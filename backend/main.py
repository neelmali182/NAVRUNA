from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from backend.routers import simulation

app = FastAPI(title="NAVRUNA — Offline Maritime AI Simulation Lab", version="4.1.2", docs_url="/docs")
app.add_middleware(CORSMiddleware, allow_origins=["http://127.0.0.1:8080", "http://localhost:8080"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])
app.include_router(simulation.router)

@app.get("/", include_in_schema=False)
async def home():
    return RedirectResponse("http://127.0.0.1:8080/")

@app.get("/api/health")
async def health():
    from model.rl.ppo import torch
    return {"status":"ok","mode":"OFFLINE_SYNTHETIC_RL","external_apis":False,"cuda":torch.cuda.is_available(),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU","message":"All navigation, land, ports and ocean conditions are generated/read locally."}
