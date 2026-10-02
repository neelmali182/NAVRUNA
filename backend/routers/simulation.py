from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional
import time
from threading import RLock
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from model.weather.simulator import WeatherSimulator

from model.simulation.fleet_simulator import FleetSimulator
from model.ai.simulation_trainer import SimulationTrainer
from model.rl.ppo import PPOTrainer
from model.navigation.geodata import load_ports

router = APIRouter(prefix="/api/simulation", tags=["offline-simulation"])

# Heavy objects are built on first use rather than at import. Constructing them
# here would import torch and rasterize the global coastline before the server can
# bind its port, which reads as "the app never started" to a deploy health probe.
SIM_LOCK = RLock()
TRAIN_SUBMIT_LOCK = RLock()
_INIT_LOCK = RLock()

_TRAINER: Optional[PPOTrainer] = None
_SIM: Optional[FleetSimulator] = None
_WEATHER: Optional[WeatherSimulator] = None
_TRAIN_POOL: Optional[ThreadPoolExecutor] = None


def get_trainer() -> PPOTrainer:
    global _TRAINER
    if _TRAINER is None:
        with _INIT_LOCK:
            if _TRAINER is None:
                _TRAINER = PPOTrainer(seed=20260922)
    return _TRAINER


def get_sim() -> FleetSimulator:
    global _SIM
    if _SIM is None:
        with _INIT_LOCK:
            if _SIM is None:
                _SIM = FleetSimulator(seed=20260922, trainer=get_trainer())
    return _SIM


def get_weather() -> WeatherSimulator:
    global _WEATHER
    if _WEATHER is None:
        with _INIT_LOCK:
            if _WEATHER is None:
                _WEATHER = WeatherSimulator(seed=20260922)
    return _WEATHER


def get_train_pool() -> ThreadPoolExecutor:
    global _TRAIN_POOL
    if _TRAIN_POOL is None:
        with _INIT_LOCK:
            if _TRAIN_POOL is None:
                _TRAIN_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="navruna-trainer")
    return _TRAIN_POOL


def _run_training(*args):
    trainer = get_trainer()
    try:
        return trainer.train(*args)
    except Exception as exc:
        trainer.training_status = "error"
        trainer.last_report = {**trainer.last_report, "error": str(exc)}
        trainer._log('training_error',f'Training failed: {exc}')
        raise


class StartRequest(BaseModel):
    seed: Optional[int] = None
    vessel_count: int = Field(60, ge=1, le=300)


class StepRequest(BaseModel):
    hours: float = Field(1.0, ge=0.25, le=12.5)


class TrainRequest(BaseModel):
    steps: int = Field(20000, ge=1024, le=500000)
    envs: int = Field(64, ge=8, le=256)
    rollout: int = Field(128, ge=32, le=512)
    learning_rate: float = Field(3e-4, gt=0, le=0.01)
    continuous: bool = True


@router.post("/start")
async def start_simulation(req: StartRequest) -> Dict[str, Any]:
    global _SIM
    seed = req.seed if req.seed is not None else int(time.time()) % 2_000_000_000
    def create_fleet():
        global _SIM
        with SIM_LOCK:
            _SIM = FleetSimulator(seed=seed, trainer=get_trainer())
            return _SIM.generate_fleet(req.vessel_count)
    return await asyncio.to_thread(create_fleet)


@router.post("/step")
async def step_simulation(req: StepRequest) -> Dict[str, Any]:
    return await asyncio.to_thread(_step_simulation, req.hours)


def _step_simulation(hours: float) -> Dict[str, Any]:
    with SIM_LOCK:
        return get_sim().step(hours)


@router.post("/train")
async def train_simulation(req: TrainRequest) -> Dict[str, Any]:
    trainer = get_trainer()
    with TRAIN_SUBMIT_LOCK:
        if trainer.training_status in ('starting', 'training'):
            return {"accepted": False, "status": "training", "message": "A training run is already active."}
        trainer.training_status = 'starting'
        try:
            future = get_train_pool().submit(_run_training, req.steps, req.envs, req.rollout, req.learning_rate, req.continuous)
        except Exception:
            trainer.training_status = 'error'
            raise
    return {"accepted": True, "status": "training", "requested_steps": req.steps, "future": id(future)}


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


@router.get("/policy")
async def policy() -> Dict[str, Any]:
    return get_trainer().analytics()

@router.get("/training")
async def training() -> Dict[str, Any]:
    return get_trainer().training_view()

@router.get("/weather")
async def weather(hours: float = 0.0, lat_step: float = 8.0, lon_step: float = 10.0) -> Dict[str, Any]:
    """Return a deterministic global weather field for the visualizer.

    This endpoint is independent of fleet generation, so the visualizer works
    immediately after startup even when no vessels have been generated.
    """
    field = get_weather().field(hours, lat_step=max(2.0, min(20.0, lat_step)), lon_step=max(2.0, min(20.0, lon_step)))
    return {"ok": True, "hours": hours, "count": len(field), "field": field}


@router.post("/reset")
async def reset_simulation() -> Dict[str, Any]:
    return await asyncio.to_thread(_reset_simulation)


def _reset_simulation() -> Dict[str, Any]:
    with SIM_LOCK:
        sim = get_sim()
        sim.reset()
        return sim.snapshot(include_routes=False)
