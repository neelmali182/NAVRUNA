# NAVRUNA architecture

## Runtime planes

```text
                 Cesium 3D Globe
                       │
             local port + land data
                       │
                       ▼
                  FastAPI API
                       │
             ┌─────────┴─────────┐
             │                   │
       Fleet Simulator      PPO Trainer
             │                   │
      vessel state         PyTorch / CUDA
             │                   │
             └─────────┬─────────┘
                       ▼
                Navigation policy
                       │
               Safety action shield
                       │
                 Ocean movement
```

## Navigation safety

The land system is intentionally independent of the policy. The policy may propose a bad action during training, but the simulator has the final authority over collision geometry. This separation is important for safe RL experiments.

- Policy: chooses among nine ordinary course changes and a recovery action.
- Fast mask: screens ordinary actions using the local raster field.
- Safety mask: validates the full course sweep with vector coastline geometry and the 5 km buffer.
- Recovery: when every ordinary action is blocked, searches wider turns and lower speeds; it is legal only if its swept path passes the same vector check.
- Environment: applies the selected safe course. If none exists, the fleet holds briefly and then records a stranded-vessel failure.

## Training curriculum

1. Short point-to-point navigation, with a hard land safety mask active throughout.
2. A randomized set of fleet cruise speeds and procedural environmental conditions.
3. Gradually longer routes, increasing the training route limit after completed episodes.
4. Recovery courses are represented in PPO rollouts rather than converted into arbitrary unsafe actions.
5. Multi-vessel extensions need their own encounter observations and policy interface.

## GPU path

The model is a small MLP by design. Performance comes from batching many environments rather than making the network unnecessarily large. PyTorch automatically selects CUDA when available.

## Weather simulation subsystem

`model/weather/simulator.py` is intentionally isolated from the navigation environment. It provides deterministic spatially and temporally coherent wind, waves, currents, storm intensity, precipitation, visibility and pressure. The live fleet uses the scenario seed and shared simulation clock for policy observations and telemetry, matching the globe's field. Standalone training and evaluation episodes use their own deterministic seeds and episode clocks.

The browser field endpoint returns an explicit regular-grid description covering both poles. The grid wraps longitude over 360 degrees and is interpolated bilinearly for pointer inspection and flow-particle sampling. Longitude influence tapers to zero at the poles, and direction angles are interpolated as vectors to avoid discontinuities near north. The heat field is rendered into a cached canvas and rebuilt after camera, field, layer, or viewport changes; flow markers are separately capped and can be disabled. The HUD reports both simulation time and the field snapshot time because the visual field is fetched at three-hour simulation intervals.

The weather model and its units are procedural assumptions, not observations or a forecast. Layer ranges are based on the implemented generator. Wave direction and period are synthetic model outputs.

## Scenario and run records

`model/scenarios.py` defines the current Pydantic scenario contract, `navruna.scenario` version 2. It records the random seed, fleet size, synthetic weather model, app version, and SHA-256 fingerprints for the port and coastline files. The API publishes the generated JSON Schema and creates templates from the local files. Fleet initialization rejects incompatible data fingerprints or a different weather model version. Version 1 seed/count files are migrated by the UI.

Each fleet start returns a run ID, UTC start time, controller at start, and the scenario inputs. The browser retains up to 180 observed samples per vessel for the current page session. Users can scrub a selected vessel's recent trace or export a run record/trace as JSON; the export is an observed-step history, not a continuous physical recording and is not retained across a browser reload unless exported.

PPO checkpoints use an atomic replacement and include optimizer state, update/episode counters, and training-run configuration. They carry both weather and navigation-action interface versions; an older action head falls back to the bootstrap controller until retrained. If an update fails, the trainer restores the latest compatible committed checkpoint before exposing the policy again. Training history and report writes are atomic JSON replacements. Training-run metadata records the random seed, requested transitions, environment count, rollout length, training cruise speeds, learning rate, app/runtime versions, and optional code revision.

## Evaluation

`model/rl/evaluation.py` compares the bootstrap heading heuristic and, when available, the PPO checkpoint over identical deterministic seed ranges. Each episode records its ports, outcome, fuel, duration, distance, clearance, and return. The summary includes sample count, failure rates, a Wilson interval for the success rate, local data fingerprints, weather model version, checkpoint/run IDs, and limitations. Results use synthetic scenarios and small samples; confidence intervals are descriptive and do not validate real-world performance.

## HTTP boundary and UI assets

The local launcher binds the API to loopback. CORS origins and trusted hostnames use explicit allowlists; remote deployments must configure both through environment variables. The globe requests its port catalog and coastline from API endpoints backed by the same files used by simulation and collision geometry. The API health endpoint is a lightweight liveness probe and distinguishes local/synthetic simulation inputs from online browser map assets. CesiumJS and OpenStreetMap imagery remain external dependencies unless separately self-hosted.

## Training observability

The PPO trainer records update-level return, success rate, policy loss, value loss, entropy, approximate KL, learning rate and environment steps/sec. The backend exposes `/api/simulation/training`, and `frontend/training.html` renders these metrics without blocking the simulation UI.

## Deferred extensibility targets

- Historical weather replay datasets
- Bathymetry/depth constraints and draft-aware routing
- Multi-agent collision avoidance
- Port arrival windows and berth constraints
- Vessel-specific fuel curves and engine models
- Named checkpoint comparison and a persistent experiment registry
- Full route-level deterministic replay across browser reloads
- Domain-randomized evaluation suite
- ONNX/TensorRT inference profile for deployment
