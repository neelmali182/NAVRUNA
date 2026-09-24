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

- Policy: chooses heading.
- Fast mask: screens actions using the local raster field.
- Safety shield: validates the selected action with continuous vector geometry.
- Environment: applies the action or returns the terminal collision reward.

## Training curriculum

1. Short point-to-point navigation in open water.
2. Land-constrained navigation.
3. Longer routes.
4. Procedural environmental disturbances.
5. Multi-vessel extensions can be added without changing the policy interface.

## GPU path

The model is a small MLP by design. Performance comes from batching many environments rather than making the network unnecessarily large. PyTorch automatically selects CUDA when available.

## Weather simulation subsystem

`model/weather/simulator.py` is intentionally isolated from the navigation environment. It provides deterministic spatially and temporally coherent wind, waves, currents, storm intensity, precipitation, visibility and pressure. The RL environment consumes the same field, so visualized conditions and training conditions share one source of truth.

## Training observability

The PPO trainer records update-level return, success rate, policy loss, value loss, entropy, approximate KL, learning rate and environment steps/sec. The backend exposes `/api/simulation/training`, and `frontend/training.html` renders these metrics without blocking the simulation UI.

## Next extensibility targets

- Historical weather replay datasets
- Bathymetry/depth constraints and draft-aware routing
- Multi-agent collision avoidance
- Port arrival windows and berth constraints
- Vessel-specific fuel curves and engine models
- Automatic checkpoint comparison / experiment registry
- Deterministic scenario replay and seed sharing
- Domain-randomized evaluation suite
- ONNX/TensorRT inference profile for deployment
