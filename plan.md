# NAVRUNA — AI-Powered Fleet Navigation System: Master Improvement Plan

> **Project**: NAVRUNA v4.2.0 → v5.0  
> **Date**: October 2026  
> **Goal**: Transform a research simulator into a polished, production-grade AI fleet navigation platform

---

## 1. Current State Assessment

### What Exists (Strengths)
| Component | Status | Notes |
|---|---|---|
| **PPO RL Training** | ✅ Working | 256-256-128 MLP, action masking, safety shield |
| **Fleet Simulator** | ✅ Working | Multi-vessel, port-to-port voyages with auto-reassignment |
| **Procedural Weather** | ✅ Working | Deterministic wind/waves/current/storm model |
| **Coastline Safety** | ✅ Working | Vector + raster dual-authority collision system |
| **3D Globe (Cesium)** | ✅ Working | Live vessel tracking, weather overlays, telemetry |
| **Training Lab** | ✅ Working | Live charts, metrics, activity log |
| **Evaluation System** | ✅ Working | Paired-seed controller comparison |
| **Scenario System** | ✅ Working | Versioned, fingerprinted, save/load |

### What's Broken / Needs Fixing
| Issue | Severity | Location |
|---|---|---|
| `model/ai/simulation_trainer.py` is a dead wrapper — `SimulationTrainer` is just `PPOTrainer` with no added logic | Low | `model/ai/` |
| No `.env` file shipped (only `.env.example`), first-time users get no guidance | Medium | Root |
| `README.md` references `bugs.md` but the file is at `docs/BUG_REPORT.md` | Low | `README.md:217` |
| No error handling in frontend JS for network failures during simulation steps | Medium | `frontend/js/simulation.js` |
| Weather field refetch is hardcoded to 3-hour buckets — large time jumps skip weather transitions | Low | `simulation.js:63` |
| `data/models/` directory is empty — no pre-trained checkpoint ships with the project | Medium | `data/models/` |
| CSS file is a single 15KB monolith with no component separation | Low | `frontend/css/styles.css` |
| No WebSocket or SSE — all updates are poll-based, causing latency | Medium | Backend/Frontend |
| Training page HTML is entirely on 6 lines — unreadable/unmaintainable | Low | `frontend/training.html` |
| Analytics page HTML is entirely on 2 lines — same problem | Low | `frontend/analytics.html` |
| No favicon or PWA manifest | Low | Frontend |
| `fleet_simulator.py` lines are extremely long (some > 300 chars), hurting readability | Medium | `model/simulation/fleet_simulator.py` |
| `geodata.py` has compressed formatting with no spaces — hard to read | Low | `model/navigation/geodata.py` |
| No type hints on many return values | Low | Throughout backend |
| `ppo.py` train method is 130+ lines — needs decomposition | Medium | `model/rl/ppo.py` |

---

## 2. Architecture Improvements

### 2.1 WebSocket Real-Time Communication
**Current**: Frontend polls via `setInterval` + REST calls  
**Proposed**: Add WebSocket endpoint for live simulation streaming

```
Backend (FastAPI)                  Frontend
     │                                │
     ├─── /ws/simulation ────────────►│  Live vessel positions
     ├─── /ws/training ──────────────►│  Training metrics stream
     └─── /ws/weather ───────────────►│  Weather field updates
```

**Changes needed:**
- Add `websockets` to `requirements.txt`
- Create `backend/routers/ws.py` with WebSocket handlers
- Modify `fleet_simulator.py` to emit events
- Update `simulation.js` to use WebSocket with REST fallback

### 2.2 Proper Configuration System
**Current**: Hardcoded values scattered across files  
**Proposed**: Centralized `config.py` with environment variable overrides

```python
# backend/config.py
class NavrunaConfig:
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000
    MAX_VESSELS: int = 300
    DEFAULT_SEED: int = 20260922
    SAFETY_BUFFER_KM: float = 5.0
    CHECKPOINT_INTERVAL: int = 25
    MAX_TRAINING_HISTORY: int = 2000
```

### 2.3 Logging System
**Current**: No structured logging — print statements and in-memory activity logs  
**Proposed**: Python `logging` with structured JSON output

### 2.4 Database Layer (Optional)
**Current**: JSON files for checkpoints, in-memory state  
**Proposed**: SQLite for experiment tracking, run history, and vessel telemetry persistence

---

## 3. Feature Additions

### 3.1 Multi-Agent Collision Avoidance (COLREGS-Inspired)
**Priority**: High — This is the biggest missing "AI" feature

- Add vessel-vessel proximity observations to the RL state vector
- Implement simplified COLREGS rules (give-way / stand-on)
- Add encounter detection and near-miss tracking
- New observation features: relative positions and velocities of nearest N vessels

**Files to modify:**
- `model/rl/navigation_env.py` — extend `obs_size` from 25 → 35+
- `model/rl/ppo.py` — update `PolicyNet` input size
- `model/simulation/fleet_simulator.py` — add inter-vessel awareness

### 3.2 Historical Weather Replay
**Priority**: Medium

- Add support for loading real weather data (ERA5 or similar)
- Keep procedural model as fallback/default
- New `WeatherProvider` interface with `ProceduralWeather` and `HistoricalWeather` implementations

### 3.3 Fuel Optimization Objective
**Priority**: High

- Add explicit fuel-efficiency reward shaping
- Track and display fuel per nautical mile as a first-class metric
- Add "eco mode" vs "speed mode" training profiles
- Vessel-class-specific fuel curves

### 3.4 Enhanced Analytics Dashboard
**Priority**: High

- Voyage replay with timeline scrubbing
- Heat maps of vessel density
- Fuel consumption charts over time
- Per-vessel performance comparison
- Fleet-wide KPI dashboard with historical trends

### 3.5 Route Visualization
**Priority**: Medium

- Draw planned vs actual route polylines on the globe
- Show deviation corridors
- Waypoint markers with ETA estimates

### 3.6 Alert System
**Priority**: Medium

- Storm warnings when vessels enter high-storm zones
- Low fuel alerts with configurable thresholds
- Near-miss notifications (vessel-vessel, vessel-land)
- Voyage completion/failure notifications

### 3.7 Pre-trained Checkpoint
**Priority**: High — Currently ships with no trained model

- Train a baseline checkpoint and bundle it in `data/models/`
- Version it with the navigation and weather model versions
- Users can immediately see AI-controlled vessels without training first

### 3.8 Batch Experiment Runner
**Priority**: Low

- Run multiple training configurations automatically
- Compare hyperparameter sweeps
- Export experiment results as structured reports

### 3.9 Docker Support
**Priority**: Medium

- Dockerfile for single-container deployment
- `docker-compose.yml` for multi-service setup
- GPU passthrough configuration

---

## 4. Code Quality & Refactoring

### 4.1 Backend Refactoring

| Task | Files | Priority |
|---|---|---|
| Decompose `PPOTrainer.train()` into `_collect_rollout()`, `_compute_advantages()`, `_update_policy()` | `model/rl/ppo.py` | High |
| Extract vessel creation logic from `FleetSimulator.generate_fleet()` | `model/simulation/fleet_simulator.py` | Medium |
| Remove dead `model/ai/` wrapper module or give it real purpose | `model/ai/` | Low |
| Add proper type annotations to all public methods | Throughout | Medium |
| Fix line length violations (PEP 8) throughout model files | `model/**/*.py` | Medium |
| Add docstrings to all classes and public methods | Throughout | Medium |
| Create `backend/services/` layer to separate business logic from route handlers | `backend/routers/simulation.py` | High |

### 4.2 Frontend Refactoring

| Task | Files | Priority |
|---|---|---|
| Format HTML files properly (they're currently single-line) | `training.html`, `analytics.html` | High |
| Split `simulation.js` (37KB monolith) into modules | `frontend/js/` | High |
| Add error boundaries and user-facing error messages | `frontend/js/*.js` | Medium |
| Create reusable chart component | `frontend/js/` | Medium |
| Add loading states and skeleton screens | Frontend | Medium |
| Responsive design improvements for mobile/tablet | `frontend/css/styles.css` | Low |

### 4.3 Testing Improvements

| Task | Priority |
|---|---|
| Add integration tests for all API endpoints | High |
| Add frontend smoke tests (Playwright/Puppeteer) | Medium |
| Test weather model edge cases (poles, date line) | Medium |
| Add training convergence regression test | Medium |
| Test scenario save/load roundtrip | Medium |
| Increase code coverage to >80% | Medium |

---

## 5. UI/UX Overhaul

### 5.1 Dark Theme Refinement
- Current theme is good but needs consistency
- Add theme variables for all colors
- Support light/dark mode toggle

### 5.2 Onboarding Experience
- First-run tutorial overlay
- Guided scenario setup
- Tooltips on all controls
- "Quick Demo" button that auto-starts a pre-configured simulation

### 5.3 Keyboard Shortcuts
| Key | Action |
|---|---|
| `Space` | Start/stop simulation |
| `S` | Step +1H |
| `R` | Reset scenario |
| `W` | Toggle weather |
| `1-4` | Switch weather layers |
| `Esc` | Deselect vessel |
| `N` | North up view |

### 5.4 Performance Indicators
- FPS counter (debug mode)
- API latency indicator
- Memory usage monitoring
- GPU utilization during training

---

## 6. Phased Implementation Roadmap

### Phase 1: Foundation, UI Overhaul & Bug Fixes (Completed / In Progress)
> Get the existing project solid, modernized, and running smoothly

- [x] Fix `README.md` broken link to `bugs.md` (updated to `docs/BUG_REPORT.md`)
- [x] Complete UI / Theme / Design overhaul to **Aero-Oceanic Aurora** deep obsidian theme
- [x] Format and redesign `training.html` and `analytics.html` with clean semantic HTML
- [x] Add proper error handling in frontend JS network calls with user toasts
- [x] Create `backend/config.py` centralized configuration with environment overrides
- [x] Wire `backend/config.py` into `backend/main.py` and `backend/routers/simulation.py`
- [x] Add SVG favicon (`favicon.svg`) with high-tech beacon iconography
- [x] Ship `.env` and add automatic bootstrap in `start.ps1`
- [x] Add keyboard shortcuts (`Space`, `S`, `R`, `N`, `1`-`4`, `Esc`)
- [x] Fix Cesium scene colors, coastline stroke/fill, and vessel telemetry colors
- [ ] Add Python `logging` throughout backend
- [ ] Fix code formatting in `geodata.py`, `fleet_simulator.py`, `ppo.py`
- [ ] Remove or repurpose `model/ai/simulation_trainer.py`
- [ ] Add comprehensive docstrings

### Phase 2: Core Architecture (Week 3-4)
> Add real-time communication and better state management

- [ ] Implement WebSocket endpoint for live simulation streaming
- [ ] Decompose `PPOTrainer.train()` into smaller methods
- [ ] Create `backend/services/` layer
- [ ] Split `simulation.js` into modules (`globe.js`, `weather.js`, `telemetry.js`, `controls.js`)
- [ ] Add loading states and better error feedback in UI
- [ ] Train and ship a baseline pre-trained checkpoint
- [ ] Add keyboard shortcuts

### Phase 3: New AI Features (Week 5-8)
> The flagship improvements that make NAVRUNA stand out

- [ ] Implement multi-vessel awareness in RL observations
- [ ] Add fuel optimization reward shaping and eco/speed profiles
- [ ] Add vessel-vessel proximity tracking and near-miss detection
- [ ] Implement route visualization (planned vs actual paths)
- [ ] Add alert system for storms, fuel, and near-misses
- [ ] Enhanced analytics dashboard with heat maps and charts
- [ ] Voyage replay with full timeline scrubbing

### Phase 4: Production Readiness (Week 9-10)
> Polish, testing, and deployment

- [ ] Full API integration test suite
- [ ] Frontend smoke tests
- [ ] Docker containerization
- [ ] Performance optimization audit
- [ ] Documentation update (README, ARCHITECTURE, MODEL_CARD)
- [ ] Cross-platform launcher (Linux/macOS support)
- [ ] Version bump to v5.0 release

---

## 7. File-Level Change Map

### New Files to Create
```
backend/config.py                    # Centralized configuration
backend/services/__init__.py         # Service layer
backend/services/simulation.py       # Simulation business logic
backend/services/training.py         # Training business logic
backend/routers/ws.py                # WebSocket endpoints
frontend/js/globe.js                 # Globe initialization and camera
frontend/js/weather.js               # Weather overlay system
frontend/js/telemetry.js             # Vessel telemetry panel
frontend/js/controls.js              # Mission control UI bindings
frontend/js/charts.js                # Reusable chart components
frontend/js/shortcuts.js             # Keyboard shortcut handler
frontend/js/ws.js                    # WebSocket client
frontend/favicon.ico                 # Site favicon
frontend/manifest.json               # PWA manifest
Dockerfile                           # Container definition
docker-compose.yml                   # Multi-service orchestration
scripts/train_baseline.py            # Script to create bundled checkpoint
```

### Files to Modify
```
backend/main.py                      # Add WebSocket mount, config
backend/routers/simulation.py        # Extract to service layer, add types
model/rl/ppo.py                      # Decompose train(), add logging
model/rl/navigation_env.py           # Multi-agent observations
model/simulation/fleet_simulator.py  # Event emission, better formatting
model/navigation/geodata.py          # Code formatting, type hints
model/weather/simulator.py           # WeatherProvider interface
frontend/index.html                  # Favicon, shortcuts, better structure
frontend/training.html               # Proper formatting
frontend/analytics.html              # Proper formatting, more charts
frontend/css/styles.css              # Theme variables, responsive
frontend/js/simulation.js            # Split into modules
frontend/js/training.js              # WebSocket support
frontend/js/analytics.js             # Enhanced charts
requirements.txt                     # Add websockets, logging deps
README.md                            # Fix links, update for v5
docs/ARCHITECTURE.md                 # Update with new components
start.ps1                            # Config file support
VERSION                              # 4.2.0 → 5.0.0
```

### Files to Delete / Deprecate
```
model/ai/simulation_trainer.py       # Dead wrapper → remove or repurpose
model/ai/__init__.py                 # Remove if module removed
```

---

## 8. Technical Decisions

| Decision | Choice | Rationale |
|---|---|---|
| **Real-time protocol** | WebSocket with REST fallback | FastAPI has native WebSocket support; REST stays for compatibility |
| **Frontend framework** | Stay vanilla JS | Project is already working; migrating to React/Vue adds complexity without clear benefit |
| **Database** | SQLite (optional) | Zero-config, embedded, sufficient for experiment tracking |
| **Multi-agent RL** | Shared policy with extended observations | Simpler than independent policies; matches existing architecture |
| **Weather provider** | Interface pattern | Allows procedural + historical without breaking existing code |
| **Docker base image** | `python:3.12-slim` | Smallest official Python image with full compatibility |

---

## 9. Success Metrics

| Metric | Current | Target |
|---|---|---|
| Test coverage | ~30% | >80% |
| API response time (simulation step) | Unknown | <200ms for 60 vessels |
| Training throughput | Unknown | >10K steps/sec on CPU |
| Frontend load time | Unknown | <2s first paint |
| Pre-trained checkpoint success rate | N/A (no checkpoint) | >60% on 500-3000nm voyages |
| Code quality (linting) | Many violations | Zero critical violations |

---

## Next Steps

**Start with Phase 1** — fix bugs and formatting first. Each phase builds on the previous one.
Skip Phase 4 Docker/cross-platform work if you only need local development.

The highest-impact changes are:
1. **Ship a pre-trained checkpoint** (users see AI navigation immediately)
2. **WebSocket streaming** (smooth real-time vessel movement)
3. **Split the frontend monolith** (maintainability)
4. **Route visualization** (makes the AI decisions visible)
