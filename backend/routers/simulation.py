from __future__ import annotations

import asyncio
import copy
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional
import time
from threading import RLock
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from model.weather.simulator import WEATHER_MODEL_VERSION, WeatherSimulator
from model.scenarios import ScenarioConfig, make_scenario, scenario_data_matches_local

from model.simulation.fleet_simulator import FleetSimulator
from model.rl.evaluation import evaluate_policy
from model.navigation.geodata import load_ports
from backend import config

router = APIRouter(prefix="/api/simulation", tags=["offline-simulation"])
DATA_DIR = config.DATA_DIR

# Heavy objects are built on first use rather than at import. Constructing them
# here would import torch and rasterize the global coastline before the server can
# bind its port, which reads as "the app never started" to a deploy health probe.
SIM_LOCK = RLock()
TRAIN_SUBMIT_LOCK = RLock()
_INIT_LOCK = RLock()

_TRAINER: Optional["PPOTrainer"] = None
_SIM: Optional[FleetSimulator] = None
_TRAIN_POOL: Optional[ThreadPoolExecutor] = None
_EVALUATION_ACTIVE = False
_SIM_STARTING = False


def get_trainer() -> "PPOTrainer":
    global _TRAINER
    if _TRAINER is None:
        with _INIT_LOCK:
            if _TRAINER is None:
                from model.rl.ppo import PPOTrainer
                _TRAINER = PPOTrainer(seed=config.DEFAULT_SEED)
    return _TRAINER


def get_sim() -> FleetSimulator:
    global _SIM
    if _SIM is None:
        with _INIT_LOCK:
            if _SIM is None:
                _SIM = FleetSimulator(seed=config.DEFAULT_SEED, trainer=get_trainer())
    return _SIM


@lru_cache(maxsize=8)
def get_weather(seed: int = config.DEFAULT_SEED) -> WeatherSimulator:
    return WeatherSimulator(seed=int(seed))


def get_train_pool() -> ThreadPoolExecutor:
    global _TRAIN_POOL
    if _TRAIN_POOL is None:
        with _INIT_LOCK:
            if _TRAIN_POOL is None:
                _TRAIN_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="navruna-trainer")
    return _TRAIN_POOL


def _run_training(*args):
    trainer = get_trainer()
    model_before = None
    optimizer_before = None
    counters_before = (trainer.total_updates, trainer.episodes, trainer.trained, trainer.checkpoint_status)
    try:
        model_before = copy.deepcopy(trainer.policy.state_dict())
        optimizer_before = copy.deepcopy(trainer.optimizer.state_dict())
        return trainer.train(*args)
    except Exception as exc:
        failed_run = trainer.training_run
        try:
            restored = trainer.restore_last_committed_checkpoint()
        except Exception:
            restored = False
        if not restored and model_before is not None and optimizer_before is not None:
            trainer.policy.load_state_dict(model_before)
            trainer.optimizer.load_state_dict(optimizer_before)
            trainer.total_updates, trainer.episodes, trainer.trained, trainer.checkpoint_status = counters_before
        trainer.training_status = "error"
        if failed_run:
            trainer.training_run = {
                **failed_run,
                "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "status": "error",
                "error": str(exc),
            }
        trainer.last_report = {**trainer.last_report, "error": str(exc)}
        trainer.last_report["training_run"] = trainer.training_run
        trainer.last_report["trained"] = trainer.trained
        trainer.last_report["checkpoint_status"] = trainer.checkpoint_status
        trainer._log('training_error',f'Training failed: {exc}')
        try:
            from model.rl.ppo import META_PATH, _atomic_write_json
            _atomic_write_json(META_PATH, trainer.last_report)
        except OSError:
            pass
        raise


class StartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # `seed` and `vessel_count` stay accepted for older API clients. New clients
    # send the complete versioned scenario object.
    scenario: Optional[ScenarioConfig] = None
    seed: Optional[int] = Field(None, ge=0, le=2_147_483_647)
    vessel_count: Optional[int] = Field(None, ge=1, le=300)

    @model_validator(mode="after")
    def prevent_mixed_scenario_inputs(self):
        if self.scenario is not None and (self.seed is not None or self.vessel_count is not None):
            raise ValueError("send scenario or legacy seed/vessel_count fields, not both")
        return self


class StepRequest(BaseModel):
    hours: float = Field(1.0, ge=0.25, le=12.5)


class TrainRequest(BaseModel):
    steps: int = Field(20000, ge=1024, le=500000)
    envs: int = Field(64, ge=8, le=256)
    rollout: int = Field(128, ge=32, le=512)
    learning_rate: float = Field(3e-4, gt=0, le=0.01)
    continuous: bool = False


class EvaluationRequest(BaseModel):
    episodes: int = Field(8, ge=1, le=32)
    seed: int = Field(20261012, ge=0, le=2_147_483_647)


@router.post("/start")
async def start_simulation(req: StartRequest) -> Dict[str, Any]:
    global _SIM, _SIM_STARTING
    if req.scenario is not None:
        config = req.scenario
        if not scenario_data_matches_local(config):
            raise HTTPException(status_code=409, detail="Scenario dataset fingerprints do not match this NAVRUNA installation.")
        if config.weather.model_version != WEATHER_MODEL_VERSION:
            raise HTTPException(status_code=409, detail=f"Scenario weather model {config.weather.model_version} is not supported by this installation ({WEATHER_MODEL_VERSION}).")
    else:
        seed = req.seed if req.seed is not None else int(time.time()) % 2_000_000_000
        config = make_scenario(seed, req.vessel_count if req.vessel_count is not None else 60)

    trainer = get_trainer()
    with TRAIN_SUBMIT_LOCK:
        if trainer.training_status in ('starting', 'training', 'stopping'):
            raise HTTPException(status_code=409, detail="Stop training before starting a fleet simulation.")
        if _SIM_STARTING:
            raise HTTPException(status_code=409, detail="A fleet is already being initialized.")
        _SIM_STARTING = True

    def create_fleet():
        global _SIM
        with SIM_LOCK:
            _SIM = FleetSimulator(seed=config.seed, trainer=trainer, scenario_config=config.model_dump(mode="json"))
            return _SIM.generate_fleet(config.vessel_count)
    try:
        return await asyncio.to_thread(create_fleet)
    finally:
        with TRAIN_SUBMIT_LOCK:
            _SIM_STARTING = False


@router.post("/step")
async def step_simulation(req: StepRequest) -> Dict[str, Any]:
    return await asyncio.to_thread(_step_simulation, req.hours)


def _step_simulation(hours: float) -> Dict[str, Any]:
    with SIM_LOCK:
        return get_sim().step(hours)


@router.post("/train")
async def train_simulation(req: TrainRequest) -> Dict[str, Any]:
    global _SIM_STARTING
    trainer = get_trainer()
    with TRAIN_SUBMIT_LOCK:
        if _SIM_STARTING:
            return {"accepted": False, "status": "starting_simulation", "message": "Wait for fleet initialization to finish before starting training."}
        if _EVALUATION_ACTIVE:
            return {"accepted": False, "status": "evaluating", "message": "Wait for the current evaluation to finish before starting training."}
        if trainer.training_status in ('starting', 'training'):
            return {"accepted": False, "status": "training", "message": "A training run is already active."}
        trainer.stop_requested.clear()
        trainer.training_status = 'starting'
        try:
            future = get_train_pool().submit(_run_training, req.steps, req.envs, req.rollout, req.learning_rate, req.continuous)
        except Exception:
            trainer.training_status = 'error'
            raise
    return {"accepted": True, "status": "training", "requested_steps": req.steps, "future": id(future)}


@router.post("/evaluate")
async def evaluate(req: EvaluationRequest) -> Dict[str, Any]:
    global _EVALUATION_ACTIVE
    trainer = get_trainer()
    with TRAIN_SUBMIT_LOCK:
        if _EVALUATION_ACTIVE:
            return {"accepted": False, "message": "An evaluation is already running."}
        if trainer.training_status in ('starting', 'training'):
            return {"accepted": False, "message": "Stop training before starting an evaluation."}
        _EVALUATION_ACTIVE = True
    try:
        result = await asyncio.to_thread(evaluate_policy, trainer, req.episodes, req.seed)
        return {"accepted": True, **result}
    finally:
        with TRAIN_SUBMIT_LOCK:
            _EVALUATION_ACTIVE = False


@router.post("/train/stop")
async def stop_training() -> Dict[str, Any]:
    trainer = get_trainer()
    if trainer.training_status not in ('starting', 'training'):
        return {"accepted": False, "status": trainer.training_status, "message": "No training run is active."}
    trainer.request_stop()
    return {"accepted": True, "status": "stopping", "message": "Training will stop after the current PPO update."}


@router.get("/state")
async def simulation_state() -> Dict[str, Any]:
    return await asyncio.to_thread(_simulation_state)


def _simulation_state() -> Dict[str, Any]:
    with SIM_LOCK:
        return get_sim().snapshot(include_routes=False)


@router.get("/analytics")
async def simulation_analytics() -> Dict[str, Any]:
    with SIM_LOCK:
        return {**get_sim().analytics(), "training": get_trainer().analytics(), "ports": len(load_ports())}


@router.get("/ports")
async def ports() -> Dict[str, Any]:
    return {"count": len(load_ports()), "ports": load_ports()}


@router.get("/coastline")
async def coastline() -> FileResponse:
    """Serve the exact coastline file used by the navigation safety geometry."""
    path = DATA_DIR / "geography" / "land.geojson"
    if not path.is_file():
        raise HTTPException(status_code=503, detail="Bundled coastline data is unavailable.")
    return FileResponse(path, media_type="application/geo+json", filename="land.geojson")


@router.get("/policy")
async def policy() -> Dict[str, Any]:
    return get_trainer().analytics()

@router.get("/training")
async def training() -> Dict[str, Any]:
    return get_trainer().training_view()

@router.get("/scenario/schema")
async def scenario_schema() -> Dict[str, Any]:
    """Expose the authoritative JSON schema for downloadable scenario files."""
    return ScenarioConfig.model_json_schema()


@router.get("/scenario/template")
async def scenario_template(
    seed: int = Query(20260922, ge=0, le=2_147_483_647),
    vessel_count: int = Query(60, ge=1, le=300),
) -> Dict[str, Any]:
    """Build a versioned scenario with fingerprints for this local data bundle."""
    return make_scenario(seed, vessel_count).model_dump(mode="json")


@router.get("/weather")
async def weather(
    hours: float = Query(0.0, ge=0.0, le=1_000_000.0),
    seed: int = Query(20260922, ge=0, le=2_147_483_647),
    lat_step: float = Query(5.0, ge=2.0, le=20.0),
    lon_step: float = Query(5.0, ge=2.0, le=20.0),
) -> Dict[str, Any]:
    """Return a deterministic global weather field for the visualizer.

    This endpoint is independent of fleet generation, so the visualizer works
    immediately after startup even when no vessels have been generated.
    """
    field, grid = get_weather(seed).field_with_grid(hours, lat_step=lat_step, lon_step=lon_step)
    return {"ok": True, "kind": "synthetic", "model_version": WEATHER_MODEL_VERSION, "seed": seed, "hours": hours, "count": len(field), "grid": grid, "field": field}


@router.post("/reset")
async def reset_simulation() -> Dict[str, Any]:
    return await asyncio.to_thread(_reset_simulation)


def _reset_simulation() -> Dict[str, Any]:
    with SIM_LOCK:
        sim = get_sim()
        sim.reset(clear_run=True)
        return sim.snapshot(include_routes=False)
