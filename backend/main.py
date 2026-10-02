import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from backend.routers import simulation

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT / "frontend"

app = FastAPI(title="NAVRUNA — Offline Maritime AI Simulation Lab", version="4.1.3", docs_url="/docs")

# The local launcher serves the UI from a second process on :8080, so the API is
# cross-origin there. On a host that serves both from one origin CORS is unused;
# set NAVRUNA_ALLOWED_ORIGINS (comma-separated) when the UI lives elsewhere.
_default_origins = "http://127.0.0.1:8080,http://localhost:8080"
_origins = [o.strip() for o in os.getenv("NAVRUNA_ALLOWED_ORIGINS", _default_origins).split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_origin_regex=r"https?://.*\.antideploy\.app",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
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
        "mode": "OFFLINE_SYNTHETIC_RL",
        "external_apis": False,
        "message": "All navigation, land, ports and ocean conditions are generated/read locally.",
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
