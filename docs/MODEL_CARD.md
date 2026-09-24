# NAVRUNA PPO model card

**Task:** synthetic maritime point-to-point navigation.

**Algorithm:** Proximal Policy Optimization (PPO).

**Observation:** 25-dimensional normalized local state.

**Action:** 7 discrete heading changes.

**Terminal rewards:** destination +2; land collision -2; fuel/time failure -2.

**Safety:** training permits collision termination; deployment uses a separate action-safety shield before execution.

**Data:** synthetic procedural environment plus bundled offline coastline and port catalog.

**Intended use:** software engineering, reinforcement-learning research, simulation and classroom demonstration.

**Not intended for:** real vessel control, navigation decisions, or safety-critical deployment.
