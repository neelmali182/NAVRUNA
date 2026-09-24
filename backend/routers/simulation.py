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

TRAINER = PPOTrainer(seed=20260922)
SIM = FleetSimulator(seed=20260922, trainer=TRAINER)
TRAIN_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="navruna-trainer")
WEATHER = WeatherSimulator(seed=20260922)
SIM_LOCK = RLock()
TRAIN_SUBMIT_LOCK = RLock()

def _run_training(*args):
    try:
        return TRAINER.train(*args)
    except Exception as exc:
        TRAINER.training_status = "error"
        TRAINER.last_report = {**TRAINER.last_report, "error": str(exc)}
        TRAINER._log('training_error',f'Training failed: {exc}')
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
    global SIM
    seed = req.seed if req.seed is not None else int(time.time()) % 2_000_000_000
    def create_fleet():
        global SIM
        with SIM_LOCK:
            SIM = FleetSimulator(seed=seed, trainer=TRAINER)
            return SIM.generate_fleet(req.vessel_count)
    return await asyncio.to_thread(create_fleet)


@router.post("/step")
async def step_simulation(req: StepRequest) -> Dict[str, Any]:
    return await asyncio.to_thread(_step_simulation, req.hours)


def _step_simulation(hours: float) -> Dict[str, Any]:
    with SIM_LOCK:
        return SIM.step(hours)


@router.post("/train")
async def train_simulation(req: TrainRequest) -> Dict[str, Any]:
    with TRAIN_SUBMIT_LOCK:
        if TRAINER.training_status in ('starting', 'training'):
            return {"accepted": False, "status": "training", "message": "A training run is already active."}
        TRAINER.training_status = 'starting'
        try:
            future = TRAIN_POOL.submit(_run_training, req.steps, req.envs, req.rollout, req.learning_rate, req.continuous)
        except Exception:
            TRAINER.training_status = 'error'
            raise
    return {"accepted": True, "status": "training", "requested_steps": req.steps, "future": id(future)}


@router.post("/train/stop")
async def stop_training() -> Dict[str, Any]:
    if TRAINER.training_status not in ('starting', 'training'):
        return {"accepted": False, "status": TRAINER.training_status, "message": "No training run is active."}
    TRAINER.request_stop()
    return {"accepted": True, "status": "stopping", "message": "Training will stop after the current PPO update."}


@router.get("/state")
async def simulation_state() -> Dict[str, Any]:
    return await asyncio.to_thread(_simulation_state)


def _simulation_state() -> Dict[str, Any]:
    with SIM_LOCK:
        return SIM.snapshot(include_routes=False)


@router.get("/analytics")
async def simulation_analytics() -> Dict[str, Any]:
    with SIM_LOCK:
        return {**SIM.analytics(), "training": TRAINER.analytics(), "ports": len(load_ports())}


@router.get("/ports")
async def ports() -> Dict[str, Any]:
    return {"count": len(load_ports()), "ports": load_ports()}


@router.get("/policy")
async def policy() -> Dict[str, Any]:
    return TRAINER.analytics()

@router.get("/training")
async def training() -> Dict[str, Any]:
    return TRAINER.training_view()

@router.get("/weather")
async def weather(hours: float = 0.0, lat_step: float = 8.0, lon_step: float = 10.0) -> Dict[str, Any]:
    """Return a deterministic global weather field for the visualizer.

    This endpoint is independent of fleet generation, so the visualizer works
    immediately after startup even when no vessels have been generated.
    """
    field = WEATHER.field(hours, lat_step=max(2.0, min(20.0, lat_step)), lon_step=max(2.0, min(20.0, lon_step)))
    return {"ok": True, "hours": hours, "count": len(field), "field": field}


@router.post("/reset")
async def reset_simulation() -> Dict[str, Any]:
    return await asyncio.to_thread(_reset_simulation)


def _reset_simulation() -> Dict[str, Any]:
    with SIM_LOCK:
        SIM.reset()
        return SIM.snapshot(include_routes=False)
