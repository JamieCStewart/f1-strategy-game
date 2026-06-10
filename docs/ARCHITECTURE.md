# Pit Wall — Architecture

> An F1-style race strategy game. You are the team strategist: two cars, one pit wall, every call is yours.
> This document is the founding design record. Decisions referenced as ADR-NNNN are recorded in [`docs/adr/`](./adr/).

**Status:** Living document. Last updated 2026-06-10.

---

## 1. Vision

A browser game in which the player controls the race strategy of both cars of one team across a full Grand Prix: pit timing, tyre compounds, pace management, team orders, and (in later phases) wet-weather calls. Rival teams are controlled by AI strategists.

The project has three deliberately distinct pillars, each held to a professional standard:

1. **A statistically rigorous race simulation**, trained on historical timing data, with quantified uncertainty. This is the core data-science artifact.
2. **An attractive, responsive frontend** that makes live strategic information legible under time pressure.
3. **Production-shaped infrastructure** — containerized, declaratively provisioned, horizontally scalable — even while the active user count is one.

A non-goal of v1: driving physics. Cars are not simulated at corner level; the unit of simulation is the lap (with sub-lap resolution only where strategy requires it, e.g. pit deltas and overtaking windows).

## 2. Design principles

- **Simulation purity.** The race engine is a pure Python library: no I/O, no network, no clocks. Time and randomness are injected. (ADR-0002)
- **Reproducibility as a feature.** Every race is fully determined by `(model_version, seed, decision_log)`. Replays, regression tests, and bug reports all fall out of this for free. (ADR-0002)
- **Uncertainty is first-class.** The lap-time model is probabilistic end-to-end; the game *is* decision-making under uncertainty, so the model must quantify it honestly. (ADR-0003)
- **Server-authoritative state.** The browser renders; it never decides. (ADR-0004)
- **Multiplayer-shaped from day one.** The "race room" is the unit of session and of scale. v1 rooms hold one human; later versions hold several. No rewrite required.
- **Cost-aware infrastructure.** Local-first development; cloud environments are ephemeral and fully reproducible from code. (ADR-0006)
- **Decisions are documented.** Every consequential choice gets an ADR. (ADR-0001)

## 3. Repository layout (monorepo)

```
pit-wall/
├── sim/          # Pure-Python race engine (no web deps, no I/O)
├── model/        # Data pipeline, training, validation, notebooks, model cards
├── api/          # FastAPI gateway: REST (lobby/setup) + WebSockets (live race)
├── workers/      # Race-room workers: tick loop, state machine, AI strategists
├── web/          # React + TypeScript frontend
├── infra/        # Terraform (GCP) + Helm charts + docker-compose for local dev
├── docs/         # This document, ADRs, model documentation
└── .github/      # CI workflows
```

`sim/` and `model/` are importable packages with their own test suites and can be demonstrated standalone (notebooks, CLI race runner) without any web stack.

## 4. Runtime architecture

```
Browser (React)
   │  WebSocket (race state ticks, decision prompts)
   │  REST (create race, lobby, replays)
   ▼
API gateway (FastAPI, stateless, N pods)
   │  Redis pub/sub (state fan-out) + Redis (session/room registry)
   ▼
Race-room worker (owns one or more rooms)
   ├── sim engine (tick loop)
   ├── AI strategists (Monte Carlo rollouts via the same sim engine)
   └── decision state machine
   ▼
Postgres (race results, replay records, later: accounts, seasons)
```

### 4.1 Race room lifecycle

A room is created via REST, assigned to a worker, and advances through a state machine:

`lobby → formation → running ⇄ decision_pending → finished`

- **running:** the worker advances the sim on a wall-clock cadence (default ~10–15 s per lap, configurable), broadcasting a `RaceStateTick` after each step.
- **decision_pending:** entered when one of the player's cars hits a decision window (pit window opening, rain crossover approaching, rival boxed against you, damage event). The clock stops; the client receives a `DecisionPrompt` containing the choices, model-derived context (e.g. predicted stint outcomes with uncertainty bands), and a soft timeout. The player's `Decision` is appended to the decision log and the clock resumes.
- AI teams never pause the clock; their strategists decide within the tick.

### 4.2 Identity and persistence (v1)

Anonymous sessions: a signed session token maps to a `player_id` stored in Redis. No auth system in v1, but `player_id` is the foreign key everywhere from day one, so accounts later are purely additive. Race results and replays are persisted to Postgres keyed by `player_id`, with a nullable `season_id` so season/career mode (v2) needs no schema migration.

### 4.3 Replays

A replay record is `(model_version, seed, decision_log)` — a few kilobytes. Replaying is re-running the deterministic sim. This is both a product feature and the backbone of the test suite.

## 5. The lap-time model (`model/`)

### 5.1 Data pipeline

Historical timing data is ingested via the FastF1 library (sessions from 2018 onward): lap times, compounds, stint structure, pit losses, weather, and track status. The ETL stage produces cleaned per-lap Parquet datasets, filtering out in/out-laps, safety-car laps, and red-flag artifacts, and engineering features such as fuel-corrected pace, tyre age, stint index, and track evolution proxies. Raw data is never committed; the pipeline is.

### 5.2 Model structure

A hierarchical Bayesian model (PyMC or NumPyro — to be settled in a dedicated ADR after a benchmark spike) of lap time, with partial pooling across:

- circuit base pace and per-circuit characteristics (pit-loss time, overtaking difficulty, SC propensity);
- car–season performance offsets;
- driver offsets (pace and consistency/variance);
- compound-specific, nonlinear degradation curves (tyre age effects);
- fuel-load effect per kilogram-lap;
- track evolution across the race;
- residual lap-to-lap noise, heteroskedastic by driver.

Companion models: a per-circuit safety-car/VSC hazard model (deployment probability per lap, duration distribution) and a weather process (rain onset/intensity as a stochastic process, with a wet-crossover sub-model deferred to the weather phase).

### 5.3 Validation — and the CI gate

The model is backtested against held-out real races: stint-level predicted vs. actual pace, calibration plots (do 80% intervals contain ~80% of laps?), and strategy-relevant summaries ("predicted undercut gain at Circuit X vs. observed"). A validation suite runs in CI; a model artifact that regresses beyond tolerance on the backtest metrics cannot be promoted. Every promoted artifact ships with a model card recording data window, metrics, and known limitations.

### 5.4 Sim ↔ model contract

The sim engine consumes a `PaceModel` interface (sample a lap time given car/driver/compound/age/fuel/track state). The trained Bayesian model implements it via posterior sampling; a trivially simple stub implements it for fast unit tests. The sim never imports training code.

## 6. The rollout engine: AI strategists and player predictions (`workers/`)

The engine carries a batch dimension from day one (ADR-0007): race state is struct-of-arrays NumPy with shape `(n_rollouts, n_cars)`, the live race is a batch of size 1, and thousands of Monte Carlo continuations advance lap-by-lap simultaneously. Two consumers share this machinery:

**AI strategists.** Rival teams choose strategies by Monte Carlo planning **through the same sim engine the game runs on**: at each decision point, roll out many simulated race continuations under candidate strategies sampled from the model posterior, score by expected finishing position (with a tunable risk appetite per team personality), and pick. Difficulty levels fall out naturally from rollout budget and risk parameters. Because the planner uses the real game model, AI opponents are exactly as good as the model is honest.

**Prediction service (ADR-0008).** The same rollouts power the player's pit-wall tooling: on each decision prompt, the worker simulates each candidate strategy ~1,000 times from the current race state and ships server-side summaries — per-future-lap lap-time and gap quantiles (p10/p25/p50/p75/p90) plus ~20 sample traces — for the frontend to render as fan charts. Raw rollouts and posterior parameters never leave the server; the player sees the same fidelity of information the default AI plans with, so their edge is judgment, not data.

Through phase 1.2 the posterior is frozen at race start: rollouts condition on live race *state* but not on live *evidence*. Phase 1.3 lifts this via posterior-draw reweighting — treating the stored draws as particles whose weights update against observed lap times — so the ensemble visibly learns today's track conditions (ADR-0011, proposed).

## 7. Frontend (`web/`)

React + TypeScript, server state via WebSocket reducer. Core views:

- **Timing tower** — live gaps, intervals, tyre/age chips, pit status.
- **Track map** — SVG circuit with animated car markers (lap-fraction interpolation between ticks).
- **Strategy view** — gap evolution chart, **ensemble fan charts** of predicted future lap times and gaps per candidate strategy (server-computed quantile bands + sample traces, ADR-0008), pit-window overlays for both your cars.
- **Decision modal** — the pause-on-decision UI: choices, each backed by its prediction ensemble (ADR-0008), consequences preview.

Visual identity, component library choice, and the design system are deferred to a frontend ADR; the bar is "looks like a broadcast graphics package, not a dashboard template."

## 8. Naming and licensing layer

Team names, driver names, and F1 branding are trademarked. The sim therefore operates on opaque car/driver IDs; display names come from a roster configuration (`ROSTER=fictional|real`). The public deployment ships fictional; real names are a private-demo flag. No F1 logos, wordmarks, or official typefaces anywhere; the product name avoids "F1"/"Formula 1". (ADR-0005) *This is risk reduction, not legal advice.*

## 9. Infrastructure and scaling (`infra/`)

### 9.1 Local-first

`docker-compose up` runs the full stack (api, worker, redis, postgres, web dev server). `kind` is used to test Kubernetes manifests locally. The overwhelming majority of development happens at zero cloud cost.

### 9.2 Cloud (GCP)

Terraform provisions a GKE Autopilot cluster, Memorystore (Redis), Cloud SQL (Postgres), Artifact Registry, and a load balancer; Helm charts deploy the services. Cloud environments are **ephemeral**: created for demo windows and load tests, destroyed afterward with `terraform destroy` (ADR-0006). A minimal always-on demo runs on Cloud Run (scale-to-zero) so the README has a live link at near-zero cost. Initial spend is covered by GCP's new-customer credits; budget alerts are configured before the first `apply`.

### 9.3 Scaling model

- API gateway pods are stateless → trivial horizontal scaling behind the load balancer.
- Race rooms are the unit of work. A worker hosts many rooms; the worker pool scales horizontally (HPA on CPU), since AI rollouts dominate compute. Room→worker assignment lives in Redis.
- One room costs a bounded, measurable amount of CPU; capacity planning is therefore "rooms per node," and a k6 load-test report in the repo states that number with receipts.
- Observability: Prometheus + Grafana (rooms active, tick latency, rollout time, WebSocket fan-out lag), structured JSON logs.

## 10. Engineering conventions

- **Git:** trunk-based with short-lived branches; all changes via PR, even solo, squash-merged with Conventional Commits messages; tagged releases with generated changelogs.
- **ADRs:** numbered, immutable once accepted; superseded rather than edited (ADR-0001).
- **CI (GitHub Actions):** lint + type-check (ruff/mypy, eslint/tsc), sim unit + property tests (seeded determinism is asserted), model validation gate (§5.3), container builds.
- **Testing pyramid:** sim engine property tests → model backtests → API contract tests → a thin E2E happy path.

## 11. Roadmap

| Phase | Scope | Exit criterion |
|---|---|---|
| **0.1 — Engine** | Pure batched sim (ADR-0007): stints, deg, fuel, pit stops, SC events; stub pace model; CLI race runner; seeded replay tests; `benchmarks/` suite | Deterministic from a seed; 1,000-rollout batch of a full race in <1 s |
| **0.2 — Model** | FastF1 ETL; hierarchical Bayesian pace model; SC hazard model; backtest + calibration suite; CI gate | Held-out race stints predicted within stated intervals; model card published |
| **0.3 — Game loop** | API + race rooms + pause-on-decision; AI strategists (MC rollouts); prediction service (ADR-0008); minimal functional UI | A human beats/loses to AI in a full GP in the browser, locally |
| **0.4 — Frontend** | Timing tower, track map, ensemble fan charts, decision UX, visual identity | "Broadcast graphics" bar met; usable on a laptop without instruction |
| **0.5 — Infra** | Terraform + Helm + GKE ephemeral env; Cloud Run demo; observability; k6 load test report | Live demo link; "N rooms/node" documented from a real load test |
| **1.0 — Release** | Polish, replays UI, fictional roster content, README/docs pass | Public, shareable, anonymous play |
| **1.1** | Team orders, driver swaps, pace-management depth | — |
| **1.2** | Weather: rain process, inter/wet crossover, drying line | — |
| **1.3** | Live model updating: posterior-draw reweighting, per-race random effects, temperature covariate (ADR-0011) | Fan charts demonstrably tighten on in-race evidence |
| **2.0** | Accounts, leaderboards, season/career mode, multiplayer rooms | — |

Each phase maps to a milestone and a tagged release; the commit history should read as this roadmap executed in order.
