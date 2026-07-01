# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### `sim/` — race engine

All `sim/` commands must be run from `sim/` (or use `cd sim &&`). The package is managed with `uv`.

```bash
# Install (editable, with dev extras)
cd sim && pip install -e ".[dev]"

# Run a race via CLI
pitwall-race --seed 42 --rollouts 1000

# Tests
cd sim && pytest                          # full suite
cd sim && pytest tests/test_engine.py    # one file
cd sim && pytest -k test_shapes          # one test

# Lint / type-check
cd sim && ruff check src/ tests/
cd sim && mypy src/

# Benchmark (validates Phase 0.1 exit criterion)
cd sim && python benchmarks/bench_engine.py
```

### `api/` — FastAPI race server (Phase 0.3)

```bash
# Install (after installing sim first)
cd sim && pip install -e .
cd api && pip install -e .

# Run dev server (port 8000)
pitwall-server
# or: uvicorn pitwall_api.app:app --reload --port 8000
```

### `web/` — React frontend (Phase 0.3)

```bash
cd web && npm install
cd web && npm run dev        # dev server at http://localhost:5173 (proxies /api and /ws to port 8000)
cd web && npm run build      # production build
```

### Full stack (local, no Docker)

```bash
# Terminal 1:
cd api && pitwall-server
# Terminal 2:
cd web && npm run dev
# Open http://localhost:5173
```

### Docker

```bash
docker compose up
# api at :8000, web at :5173
```

## Architecture

This is a monorepo. Only `sim/` exists so far; `model/`, `api/`, `workers/`, `web/`, and `infra/` are planned (see `docs/ARCHITECTURE.md` and the roadmap).

### `sim/` — the pure race engine (Phase 0.1)

**Invariant:** `sim/` is a pure Python library — no I/O, no network, no clocks, no global state. All randomness flows through a single injected `np.random.Generator`. A race outcome is a deterministic function of `(model_version, seed, decision_log)` (ADR-0002).

**Batch dimension is fundamental (ADR-0007).** Every per-car array has shape `(n_rollouts, n_cars)`. The live race is a batch of size 1. AI rollouts and the player-facing prediction service (ADR-0008) are batches of thousands. This is not an optimization bolted on later — it is the core data layout.

**Module responsibilities:**

| File | Role |
|---|---|
| `config.py` | Frozen dataclasses: `RaceConfig`, `CircuitConfig`, `CarConfig`, `FuelConfig`, `CompoundParams`. Pure data, no randomness. |
| `state.py` | `RaceState` — struct-of-arrays with shape `(n_rollouts, n_cars)`. Mutated in-place each lap. `tiled()` broadcasts a live (R=1) state to a rollout batch. |
| `pace.py` | `PaceModel` protocol + `StubPaceModel`. The engine only calls `sample_lap_times(state, rng) -> (R, C)`. Phase 0.2 will drop in a Bayesian posterior behind the same protocol without touching the engine. |
| `strategy.py` | `StrategyPlan` (start compound + pit stop list) and `validate_strategies`. In Phase 0.3 this evolves into live `Decision` messages. |
| `engine.py` | `run_race()` (the top-level pure function) and `advance_one_lap()` (the shared step kernel used by `run_race`, planned `LiveRace`, and `continue_race`). `RaceResult` holds `(R, L, C)` lap times. |
| `defaults.py` | 20-car, Barcelona-ish config and a medium-hard two-stopper strategy for each car. Used by the CLI and benchmarks. |
| `cli.py` | Thin argparse wrapper around `run_race`; prints a timing sheet. |

**Data flow:** `run_race(config, strategies, n_rollouts, seed)` → builds a `RaceState`, iterates `advance_one_lap` L times, returns `RaceResult`. Each `advance_one_lap` call: (1) samples SC trigger, (2) calls `model.sample_lap_times`, (3) applies pit stops via boolean mask, (4) mutates state arrays in-place.

### Planned components (not yet built)

- `model/` — FastF1 ETL, hierarchical Bayesian lap-time model (PyMC or NumPyro), SC hazard model, validation suite.
- `api/` — FastAPI gateway: REST for lobby/setup, WebSockets for live race ticks.
- `workers/` — race-room workers: tick loop, AI Monte Carlo planning, player prediction service (ADR-0008).
- `web/` — React + TypeScript; timing tower, track map, ensemble fan charts, decision modal.
- `infra/` — Terraform (GCP), Helm, docker-compose for local full-stack.

### Key design decisions (ADRs in `docs/`)

- **ADR-0002** Seeded RNG; `import random`/`time`/`datetime` are banned in `sim/`. CI should enforce this.
- **ADR-0003** Hierarchical Bayesian lap-time model. The `PaceModel` protocol in `pace.py` is the contract — the trained model implements it without changing the engine.
- **ADR-0005** Roster config layer — sim operates on opaque car/driver IDs; display names come from a `ROSTER=fictional|real` flag. No F1 trademarks anywhere.
- **ADR-0007** Vectorized batch sim (see above).
- **ADR-0008** Player-facing prediction service: server ships quantile summaries (p10–p90) + ~20 sample traces; raw rollouts never leave the server.

### Testing conventions

- `_quiet_config()` in `test_engine.py` is the zero-noise, zero-SC fixture for exact assertions. Use it when a test needs deterministic lap times.
- `test_determinism.py` asserts seed reproducibility — any new source of randomness in `sim/` must be routed through the injected RNG.
- The benchmark (`benchmarks/bench_engine.py`) is the performance contract; the Phase 0.1 exit criterion is 1,000 rollouts of a 57-lap, 20-car race in < 1 s.
