# ADR-0007: Vectorized batch simulation over compiled extensions

## Context

The live race is computationally trivial (~60 laps × 20 cars of simple arithmetic), but Monte Carlo rollouts are not: an AI strategist (and, per ADR-0008, the player-facing prediction service) may evaluate several candidate strategies × ~1,000 rollouts at a single decision point — thousands of full races that must complete within a tick. A naive scalar Python loop is ~100× too slow for this. Cython, Numba, JAX, and Rust extensions were considered.

## Decision

The engine is designed around a **batch dimension from day one**:

- Race state is laid out struct-of-arrays in NumPy: `tyre_age`, `fuel`, `cumulative_time`, etc. have shape `(n_rollouts, n_cars)`. One lap-step advances *all* rollouts simultaneously; divergent per-rollout decisions (e.g. different pit laps) are applied via boolean masks.
- The live race is simply a batch of size 1 — one engine, no parallel "fast" implementation to keep in sync.
- Posterior draws from the pace model are pre-sampled into arrays per race, so the inner loop is pure array math with no inference framework on the hot path.
- Reproducibility (ADR-0002) extends to batches via `numpy.random.SeedSequence.spawn`: a batch is fully determined by its root seed.

Compiled optimizations are **deferred behind a benchmark gate**: a `benchmarks/` suite (races/sec, rollouts/decision latency) is committed in Phase 0.1, and no optimization lands without a benchmark demonstrating the need. The escalation ladder, in order of added complexity: Numba `@njit` on the step kernel → JAX (synergizes with a NumPyro model; enables GPU rollouts) → Cython/Rust as last resort.

## Consequences

- Expected 100–1000× over scalar Python with zero build-toolchain complexity in CI or Docker; the engine remains readable showcase code.
- The batch dimension slightly complicates the step function (masking) — accepted, since rollouts are a core product feature, not an afterthought.
- Every performance claim in the README is backed by a committed benchmark.
