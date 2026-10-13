import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from backend import config
from backend.routers import simulation

ROOT = config.ROOT
FRONTEND_DIR = config.FRONTEND_DIR

APP_VERSION = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
app = FastAPI(title="NAVRUNA — Maritime Research Simulator", version=APP_VERSION, docs_url="/docs")

app.add_middleware(TrustedHostMiddleware, allowed_hosts=config.ALLOWED_HOSTS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.include_router(simulation.router)


@app.get("/api/health")
async def health():
    """Liveness probe. Deliberately imports nothing heavy.

    This endpoint is what a host polls to decide the app started, and it is hit
    before the model is ever needed. Importing torch here made the answer depend
    on a 20s+ import, so the probe timed out on a perfectly healthy process.
    """
    return {
        "status": "ok",
        "mode": "LOCAL_SYNTHETIC_RESEARCH",
        "app_version": APP_VERSION,
        "simulation_external_feeds_required": False,
        "browser_map_requires_internet": True,
        "message": "Simulation inputs are local and synthetic. CesiumJS and OpenStreetMap imagery are loaded by the browser from external providers.",
    }


@app.get("/", include_in_schema=False)
async def home():
    """Send visitors to the simulator UI.

    Served as a redirect rather than a file response so the static mount below
    keeps ownership of serving assets.
    """
    return RedirectResponse("/index.html")


# Mounted last so every /api route above wins. html=True makes the mount serve
# index.html for "/" and for directory paths.
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
