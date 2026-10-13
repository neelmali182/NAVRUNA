"""Deterministic, paired-seed evaluation for navigation policy comparisons."""
from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np

from model.navigation.geodata import get_global_geography
from model.rl.navigation_env import NAVIGATION_MODEL_VERSION, OceanNavigationEnv, bootstrap_heading_action, haversine_nm
from model.scenarios import app_version, dataset_fingerprints
from model.weather.simulator import WEATHER_MODEL_VERSION


def _wilson_interval(successes: int, samples: int, z: float = 1.96) -> list[float]:
    """Wilson 95% interval for a binomial success rate."""
    if samples <= 0:
        return [0.0, 0.0]
    p = successes / samples
    z2 = z * z
    denominator = 1.0 + z2 / samples
    center = (p + z2 / (2 * samples)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z2 / (4 * samples)) / samples) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _run_episode(seed: int, controller: str, trainer):
    env = OceanNavigationEnv(seed=seed, geography=get_global_geography())
    env.reset(min_route_nm=500, max_route_nm=3000)
    env.land_enabled = True
    origin = env.origin.get("name") if isinstance(env.origin, dict) else None
    destination = env.destination.get("name") if isinstance(env.destination, dict) else None
    initial_distance = float(env.initial_distance)
    total_reward = 0.0
    travelled_nm = 0.0
    no_safe_streak = 0
    recovery_count = 0

    def finish(success: bool, failure: str = ""):
        fuel_used = max(0.0, float(env.fuel_capacity - env.fuel))
        return {
            "seed": seed,
            "controller": controller,
            "origin": origin,
            "destination": destination,
            "success": success,
            "failure": failure,
            "duration_hours": float(env.steps),
            "fuel_used_t": fuel_used,
            "fuel_per_nm_t": fuel_used / max(1.0, travelled_nm),
            "distance_travelled_nm": travelled_nm,
            "great_circle_stretch": travelled_nm / max(1.0, initial_distance),
            "minimum_clearance_km": float(env.minimum_clearance_km),
            "safe_recoveries": recovery_count,
            "return": total_reward,
        }

    while not env.done and env.steps < env.max_steps:
        obs = env.observe()
        safe = env.safe_actions(5.0)
        if not safe:
            maneuver=env.safe_recovery_maneuver(1.0,5.0)
            if maneuver is None:
                _,reward,done,info=env.hold(1.0)
                no_safe_streak+=1
            else:
                no_safe_streak=0
                previous=(env.lat,env.lon)
                _,reward,done,info=env.step(-1,1.0,turn_degrees=maneuver['turn_degrees'],speed_kn=maneuver['speed_kn'])
                travelled_nm+=haversine_nm(previous,(env.lat,env.lon))
                recovery_count+=1
            total_reward+=float(reward)
            if done:
                return finish(bool(info.get("success")),str(info.get("failure","")))
            if no_safe_streak>=3:
                return finish(False,"no_safe_action")
            continue
        no_safe_streak=0

        if controller == "PPO checkpoint":
            logits = trainer.action_logits(obs)
            action = max(safe, key=lambda idx: float(logits[idx]))
        else:
            action = bootstrap_heading_action(env,safe)

        previous = (env.lat, env.lon)
        _, reward, done, info = env.step(action, 1.0)
        recovery_count+=int(bool(info.get("recovery")))
        travelled_nm += haversine_nm(previous, (env.lat, env.lon))
        total_reward += float(reward)
        if done:
            return finish(bool(info.get("success")), str(info.get("failure", "")))

    return finish(False, "step_limit")


def _mean(results: list[dict], key: str, *, successful_only: bool = False):
    selected = [item[key] for item in results if item.get(key) is not None and (not successful_only or item["success"])]
    return float(np.mean(selected)) if selected else None


def evaluate_policy(trainer, episodes: int = 4, seed: int = 20261012):
    """Compare available controllers on the same held-out, deterministic seeds."""
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    evaluation_id = str(uuid4())
    reports = {}
    controllers = ["bootstrap heading heuristic"]
    if trainer.trained:
        controllers.append("PPO checkpoint")

    for controller in controllers:
        results = [_run_episode(seed + index, controller, trainer) for index in range(episodes)]
        successes = sum(bool(item["success"]) for item in results)
        failures = sorted({item["failure"] for item in results if item["failure"]})
        mean_fuel = _mean(results, "fuel_used_t")
        mean_duration = _mean(results, "duration_hours")
        mean_fuel_per_nm = _mean(results, "fuel_per_nm_t")
        mean_stretch = _mean(results, "great_circle_stretch", successful_only=True)
        reports[controller] = {
            "sample_count": len(results),
            "successes": successes,
            "success_rate": successes / max(1, len(results)),
            "success_rate_wilson_95_ci": _wilson_interval(successes, len(results)),
            "mean_fuel_used_t": mean_fuel,
            "mean_fuel_per_nm_t": mean_fuel_per_nm,
            "mean_duration_hours": mean_duration,
            "mean_distance_travelled_nm": _mean(results, "distance_travelled_nm"),
            "mean_return": _mean(results, "return"),
            "collision_rate": sum(item["failure"] == "land_collision" for item in results) / max(1, len(results)),
            "fuel_failure_rate": sum(item["failure"] == "fuel_exhausted" for item in results) / max(1, len(results)),
            "timeout_rate": sum(item["failure"] == "time_limit" for item in results) / max(1, len(results)),
            "no_safe_action_rate": sum(item["failure"] == "no_safe_action" for item in results) / max(1, len(results)),
            "mean_safe_recoveries": _mean(results, "safe_recoveries"),
            "mean_minimum_clearance_km": _mean(results, "minimum_clearance_km"),
            "mean_great_circle_stretch_successful_only": mean_stretch,
            "failure_counts": {reason: sum(item["failure"] == reason for item in results) for reason in failures},
            # Keep the earlier public names while clients migrate to explicit units.
            "mean_fuel_used": mean_fuel,
            "mean_steps": mean_duration,
            "mean_fuel_per_nm": mean_fuel_per_nm,
            "mean_great_circle_stretch": mean_stretch,
            "failures": {reason: sum(item["failure"] == reason for item in results) for reason in failures},
            "runs": results,
        }

    return {
        "evaluation_type": "fixed-seed paired controller comparison",
        "run_metadata": {
            "evaluation_id": evaluation_id,
            "started_at_utc": started_at,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "app_version": app_version(),
            "code_revision": os.getenv("NAVRUNA_GIT_REVISION"),
            "weather_model_version": WEATHER_MODEL_VERSION,
            "navigation_model_version": NAVIGATION_MODEL_VERSION,
            "dataset_sha256": dataset_fingerprints(),
            "seed_scheme": "seed_start + episode index",
            "checkpoint_path": trainer.last_report.get("model_path") if trainer.trained else None,
            "training_run_id": (trainer.training_run or {}).get("run_id") if trainer.trained else None,
        },
        "seed_start": seed,
        "scenario": "held-out seeded port-to-port voyages, 500–3000 nm requested distance",
        "episode_count": episodes,
        "trained_checkpoint_available": bool(trainer.trained),
        "limitations": [
            "Small deterministic sample; confidence intervals are descriptive and may be wide.",
            "Synthetic environment and weather; results do not establish real-world navigation safety.",
            "The direct-heading baseline uses the same land safety action shield as the PPO controller.",
        ],
        "controllers": reports,
    }
