# NAVRUNA PPO model card

**Task:** synthetic maritime point-to-point navigation.

**Algorithm:** Proximal Policy Optimization (PPO).

**Observation:** 25-dimensional normalized local state.

**Action:** 9 discrete course changes plus a learned recovery action. Fleet runs use the assigned ship-class cruise speed; training spans the fleet speed range.

**Terminal rewards:** destination +2; land collision -2; fuel/time failure -2.

**Safety:** PPO, fleet inference, and evaluation use the same hard action mask. Ordinary turns and the recovery course are checked against vector coastline geometry and the 5 km land buffer before movement.

**Data:** synthetic procedural environment plus bundled offline coastline and port catalog.

**Training reproducibility:** checkpoints record optimizer state, counters, and the training seed, transition target, environment count, rollout length, learning rate, runtime versions, and optional code revision. Scenario files record the app version and SHA-256 fingerprints for the port and coastline files.

**Evaluation:** paired fixed-seed voyages compare the bootstrap steering baseline with an available checkpoint. The report includes per-run outcomes, failure causes, sample count, and a Wilson interval for success rate. The default evaluation is a small research smoke benchmark, not a statistically powered qualification study.

**Weather visualization:** procedural fields have explicit units and a named model version. The visualizer interpolates a coarse global grid and reports its field time separately from the simulation clock. Weather values are synthetic and are not forecasts.

**Checkpoint compatibility:** a PPO checkpoint must carry both the current weather-model version and navigation action-interface version to control the fleet. Older weights are retained on disk but fall back to the bootstrap controller until retraining.

**Known limitations:** vessel maneuvering and fuel physics remain simplified; the local recovery search is not a global route planner. The current application does not include validated bathymetry, coral reef data, historical weather, encounter/COLREGS validation, or real-world safety evidence. A newly trained checkpoint is required after the action-interface change. Browser CesiumJS and OpenStreetMap assets are externally hosted by default.

**Intended use:** software engineering, reinforcement-learning research, simulation and classroom demonstration.

**Not intended for:** real vessel control, navigation decisions, or safety-critical deployment.
