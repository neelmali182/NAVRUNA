"""Validated, portable scenario inputs and local dataset fingerprints."""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from model.weather.simulator import WEATHER_MODEL_VERSION

ROOT = Path(__file__).resolve().parents[1]
SCENARIO_FORMAT = "navruna.scenario"
SCENARIO_VERSION = 2


@lru_cache(maxsize=1)
def dataset_fingerprints() -> dict[str, str]:
    """Fingerprint the datasets that affect scenario initialization."""
    paths = {
        "ports": ROOT / "data" / "ports" / "ports.json",
        "coastline": ROOT / "data" / "geography" / "land.geojson",
    }
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "missing"
        for name, path in paths.items()
    }


def app_version() -> str:
    version_file = ROOT / "VERSION"
    return version_file.read_text(encoding="utf-8").strip() if version_file.is_file() else "unknown"


class WeatherScenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["procedural"] = "procedural"
    seed: int = Field(ge=0, le=2_147_483_647)
    model_version: str = WEATHER_MODEL_VERSION


class DatasetFingerprints(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ports: str = Field(pattern=r"^[a-f0-9]{64}$")
    coastline: str = Field(pattern=r"^[a-f0-9]{64}$")


class ScenarioEnvironment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    app_version: str = Field(min_length=1, max_length=40)
    datasets: DatasetFingerprints


class ScenarioConfig(BaseModel):
    """Inputs currently supported by fleet initialization, with provenance."""

    model_config = ConfigDict(extra="forbid")

    format: Literal["navruna.scenario"] = SCENARIO_FORMAT
    version: Literal[2] = SCENARIO_VERSION
    seed: int = Field(ge=0, le=2_147_483_647)
    vessel_count: int = Field(ge=1, le=300)
    weather: WeatherScenario
    environment: ScenarioEnvironment

    @model_validator(mode="after")
    def weather_seed_matches_scenario(self):
        if self.weather.seed != self.seed:
            raise ValueError("weather.seed must match seed; independent weather seeds are not supported yet")
        return self


def make_scenario(seed: int, vessel_count: int) -> ScenarioConfig:
    """Construct a complete v2 scenario from the controls supported in the UI."""
    return ScenarioConfig(
        seed=seed,
        vessel_count=vessel_count,
        weather=WeatherScenario(seed=seed),
        environment=ScenarioEnvironment(app_version=app_version(), datasets=dataset_fingerprints()),
    )


def scenario_data_matches_local(scenario: ScenarioConfig) -> bool:
    return scenario.environment.datasets.model_dump() == dataset_fingerprints()
