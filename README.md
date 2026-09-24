<<<<<<< HEAD
# NAVRUNA v4 — Offline Maritime AI Simulation Lab

NAVRUNA is an offline-first research simulator in which vessels are **not given a pre-computed route**. Each vessel receives an origin, destination, fuel budget and local synthetic ocean state. A shared PPO navigation policy chooses heading actions step-by-step.

## What changed

- Replaced predefined route polylines with autonomous navigation decisions.
- Added a 279-location curated global port catalog in `data/ports/ports.json`.
- Replaced the coarse hand-written coastline with an offline GSHHS intermediate-resolution land dataset in `data/geography/land.geojson`.
- Added a raster land-distance/safety field for fast RL observations.
- Added continuous swept-segment land collision detection with a configurable 5 km safety buffer.
- Added a deployment safety shield: PPO actions are screened before movement and unsafe actions are not executed.
- Added actual PyTorch PPO training with parallel environments, curriculum learning and action masking.
- Added procedural wind, wave, current and storm fields; no live feeds are required.
- Added vessel fuel budgets and terminal failure on fuel exhaustion/time limit.
- Added RTX/CUDA detection and GPU-compatible policy training. The training code uses the CUDA device automatically when PyTorch exposes it.
- Removed route drawing from the UI. Ships move independently; the globe shows only vessel state and ports by default.
- Reworked the Cesium globe UI with modern glass panels, hideable HUDs, cinematic mode and vessel telemetry.
- Added local port and land assets to the frontend so simulation data is available without an API call.

## RL formulation

### Observation

The policy receives a 25-value local state containing:

- ship position and heading
- relative bearing/distance to destination
- remaining fuel
- speed
- synthetic wind, waves and current
- storm intensity
- seven directional land-clearance probes

### Actions

Seven discrete heading changes:

`-35°, -20°, -10°, 0°, +10°, +20°, +35°`

### Reward

Terminal rewards:

- destination reached: **+2**
- land collision: **-2**
- fuel exhausted/time limit: **-2**

Shaping rewards provide small progress/fuel/land-clearance signals between terminal events.

## Land collision architecture

The collision system intentionally does not rely on checking only the ship's current point.

1. Offline coastline polygons are loaded from GSHHS-derived GeoJSON.
2. A 0.25° occupancy/distance field provides fast RL screening.
3. Every movement is treated as a swept segment.
4. The segment is sampled continuously and checked against the vector coastline and safety field.
5. A 5 km configurable safety buffer prevents the vessel from entering land-adjacent water in the simulation.
6. During deployment, the same geometry is used as a safety shield before executing the learned action.

This is a simulation guarantee at the resolution/accuracy of the bundled geographic dataset; it is not a claim of real-world nautical safety.

## Training

The backend uses:

- PyTorch PPO
- shared policy network: `256 → 256 → 128`
- parallel environments
- curriculum learning (open-water point-to-point control first, land-constrained navigation second)
- action masking
- CUDA when available

Use **TRAIN NAVIGATION POLICY** from the UI. On your RTX 4050, a CUDA-enabled PyTorch installation will be selected automatically.

The included baseline is intentionally trainable rather than pretending that a small local checkpoint is production maritime intelligence. For demonstrations, train a longer run after installation and inspect the resulting checkpoint in `data/models/`.

## Running on Windows

1. Install Python 3.11–3.13.
2. Run `run_navruna.bat`.
3. The launcher creates `.venv`, installs requirements, starts the backend on port 8000 and frontend on port 8080.
4. Open `http://127.0.0.1:8080/`.

API docs: `http://127.0.0.1:8000/docs`

## Offline note

All **simulation data and AI environment data** are local. The visual globe currently uses CesiumJS and its Natural Earth imagery asset from the Cesium CDN. For a fully air-gapped installation, self-host the CesiumJS distribution and its imagery assets under `frontend/vendor/` and replace the two CDN references in `frontend/index.html`.

## Project structure

```text
NAVRUNA/
├── backend/
│   ├── main.py
│   └── routers/simulation.py
├── data/
│   ├── geography/land.geojson
│   ├── ports/ports.json
│   └── models/
├── frontend/
│   ├── index.html
│   ├── analytics.html
│   ├── css/
│   ├── js/
│   └── data/
├── model/
│   ├── navigation/geodata.py
│   ├── rl/navigation_env.py
│   ├── rl/ppo.py
│   └── simulation/fleet_simulator.py
├── tests/
├── requirements.txt
├── pyproject.toml
└── run_navruna.bat
```

## Validation

Run:

```bash
pytest -q
```

The test suite covers the port catalog, vector land collision authority, RL observation/network dimensions and the no-preplanned-route fleet architecture.

## Research boundary

NAVRUNA is a synthetic simulation and software engineering project. It is **not** a certified navigation system and should not be used to control real vessels. Real-world deployment would require validated nautical charts, bathymetry, COLREGS-aware behavior, AIS/weather/ocean data, vessel dynamics, uncertainty modeling, verification/validation and human oversight.

## v4.1 additions

- `model/weather/` is the dedicated offline weather/ocean simulation subsystem.
- Procedural wind, waves, currents, storms, precipitation, visibility and pressure are spatially coherent and time-varying.
- The simulator exposes a weather visualizer toggle (`W` hotkey) with wind vectors and storm intensity overlay.
- Ships render using the bundled `frontend/assets/ship.svg` image asset instead of primitive ellipsoids.
- `/api/simulation/training` exposes PPO training telemetry and history.
- `/api/simulation/train` starts training in a background worker so the Training Lab can monitor it live.
- `frontend/training.html` provides model-training telemetry, loss/return/success/entropy/KL graphs, hyperparameters and CUDA/device status.

## v4.1.1 vessel visualization
- Replaced the generic SVG vessel marker with the supplied top-down container ship image.
- Vessel icons now rotate with the learned heading, preserve the original image colors, and scale with camera distance.
- Status is shown with a small point indicator rather than tinting the ship image.


## v4.1.2 weather visualizer + launcher fixes

- Windy-inspired ocean weather visualization with smooth color-field gradients, animated particles, directional streamlines and selectable Waves/Wind/Current/Storm layers.
- Weather API is independent of fleet generation and is available immediately after backend startup.
- Weather field resolution increased for smoother global visualization.
- Startup launcher now prefers a working active Python/Conda environment and avoids unnecessary Windows `ensurepip` virtual-environment hangs.
- OpenStreetMap is used for the browser basemap to avoid the missing-tile block artifacts from the previous Natural Earth asset provider. Simulation and weather data remain synthetic/offline.
=======
# NAVRUNA 🚢

**Offline Maritime AI Simulation Lab**

NAVRUNA is an AI-powered maritime simulation platform for testing and developing intelligent vessel route optimization using simulated environments.

## Features

* 🚢 Maritime vessel simulation
* 🗺️ Interactive 3D map
* 🤖 Reinforcement Learning-based navigation
* 🌊 Simulated wind and wave conditions
* 🧭 Route optimization
* ⛽ Fuel and time optimization
* 🌱 Carbon emission estimation
* 📊 Training and simulation metrics
* 💻 Offline simulation environment

## Tech Stack

* Python
* PyTorch
* Reinforcement Learning
* CesiumJS
* JavaScript / HTML / CSS

## How to Use

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/NAVRUNA.git
cd NAVRUNA
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Start NAVRUNA

**Windows:**

```powershell
.\start.ps1
```

or:

```bash
run_navruna.bat
```

### 4. Open the Application

Open the local URL shown in the terminal, usually:

```text
http://localhost:8000
```

### 5. Run a Simulation

1. Open the NAVRUNA dashboard.
2. Select **Simulation**.
3. Select a vessel and route.
4. Start the simulation.
5. Observe the vessel navigation and route performance.
6. Use the training section to experiment with the Reinforcement Learning model.

## Project Goal

To develop an intelligent maritime navigation system that can find efficient routes while reducing **travel time, fuel consumption, and carbon emissions**.

## Status

🚧 **Under Development**
>>>>>>> 4b730e0a3b3374da292cca306468ea1ce34053bb
