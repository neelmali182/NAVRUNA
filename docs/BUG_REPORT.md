# NAVRUNA Bug Review Report

Date: 2026-09-23

## Result

The project is functional after one concurrency fix. The existing regression suite and added regression coverage pass.

## Bug Fixed

### Duplicate training jobs accepted during startup

**Severity:** Medium

**Location:** `backend/routers/simulation.py`

**Problem:** `/api/simulation/train` checked `TRAINER.training_status` before submitting work to the single-worker executor. Two requests arriving before the worker started could both observe `idle` and both be accepted. The second job would wait in the executor queue, while the API reported that training was active for both requests.

**Fix:** Added an atomic submission lock and a `starting` reservation state. A second request is rejected during both startup and active training. The stop endpoint also recognizes the startup state.

**Regression test:** `test_training_submission_is_reserved_before_worker_starts` in `tests/test_core.py`.

## Checks Performed

- `python -m pytest -q`: **8 passed**
- `python -m compileall -q backend model tests`: **passed**
- Node syntax validation for every `frontend/**/*.js`: **passed**
- FastAPI smoke checks:
  - `/api/health`: 200
  - `/api/simulation/weather`: 200
  - `/api/simulation/start`: 200
  - `/api/simulation/step`: 200
  - `/api/simulation/state`: 200
- Workspace diagnostics: no errors reported.

## Dashboard and Vessel-State Fixes

- The dashboard success tile now shows cumulative completed voyage legs (`voyages_completed`). Successful vessels are immediately assigned another voyage, so counting only vessels with `status == completed` incorrectly displayed zero.
- The bottom green progress line now advances using completed voyage legs plus failed vessels, rather than only terminal vessel statuses. It is a voyage outcome indicator, not fuel or geographic distance.
- `HOLD / REPLAN` now becomes a failed vessel with failure reason `NO SAFE ACTION`, increments voyage failures, and appears as a larger red map dot.
- Action selection now uses the authoritative swept-segment safety check across all headings when the fast mask is overly conservative. This prevents false mass failures without weakening the 5 km collision buffer.

## Non-blocking Findings

- The API smoke test reports a Starlette deprecation warning recommending `httpx2`; this did not break behavior.
- Git whitespace validation could not run because the workspace is not a Git repository.
- The review was static plus automated smoke/regression testing; no browser interaction test was run against the Cesium UI.

## Scope Notes

The training UI intentionally sends `continuous: true` in its training flows, and its interface states that continuous training runs until Stop. The `steps` value is therefore a per-update target in that mode, not an automatic stop condition; this was left unchanged.
