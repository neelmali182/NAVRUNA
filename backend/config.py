"""Centralized NAVRUNA configuration with environment variable overrides.

All tunable constants live here so they can be managed in one place
and overridden through environment variables for deployments.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ── Network ──────────────────────────────────────────────────────────
API_HOST: str = os.getenv("NAVRUNA_API_HOST", "127.0.0.1")
API_PORT: int = int(os.getenv("NAVRUNA_API_PORT", "8000"))
FRONTEND_PORT: int = int(os.getenv("NAVRUNA_FRONTEND_PORT", "8080"))

# ── CORS / Trusted Hosts ────────────────────────────────────────────
DEFAULT_ORIGINS = f"http://127.0.0.1:{FRONTEND_PORT},http://localhost:{FRONTEND_PORT}"
ALLOWED_ORIGINS: list[str] = [
    o.strip()
    for o in os.getenv("NAVRUNA_ALLOWED_ORIGINS", DEFAULT_ORIGINS).split(",")
    if o.strip()
]
DEFAULT_HOSTS = "127.0.0.1,localhost,testserver"
ALLOWED_HOSTS: list[str] = [
    h.strip()
    for h in os.getenv("NAVRUNA_ALLOWED_HOSTS", DEFAULT_HOSTS).split(",")
    if h.strip()
]

# ── Simulation ───────────────────────────────────────────────────────
MAX_VESSELS: int = int(os.getenv("NAVRUNA_MAX_VESSELS", "300"))
DEFAULT_VESSEL_COUNT: int = int(os.getenv("NAVRUNA_DEFAULT_VESSELS", "60"))
DEFAULT_SEED: int = int(os.getenv("NAVRUNA_DEFAULT_SEED", "20260922"))
SAFETY_BUFFER_KM: float = float(os.getenv("NAVRUNA_SAFETY_KM", "5.0"))

# ── Training ─────────────────────────────────────────────────────────
CHECKPOINT_INTERVAL: int = int(os.getenv("NAVRUNA_CHECKPOINT_INTERVAL", "25"))
MAX_TRAINING_HISTORY: int = int(os.getenv("NAVRUNA_MAX_HISTORY", "2000"))
MAX_ACTIVITY_LOG: int = int(os.getenv("NAVRUNA_MAX_LOG", "300"))

# ── Paths ────────────────────────────────────────────────────────────
DATA_DIR: Path = ROOT / "data"
MODEL_DIR: Path = DATA_DIR / "models"
FRONTEND_DIR: Path = ROOT / "frontend"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
