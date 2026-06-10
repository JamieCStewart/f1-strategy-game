# pitwall-sim

The pure simulation core of Pit Wall (Phase 0.1). A race is a deterministic
function of `(config, strategies, seed)` — no I/O, no clocks, no global state
(ADR-0002) — computed as vectorized array math over a batch of Monte Carlo
rollouts (ADR-0007). The live game is a batch of size 1; the AI strategists
and the player-facing prediction service (ADR-0008) are batches of thousands.

## Install & run

```bash
pip install -e ".[dev]"
pitwall-race --seed 42 --rollouts 1000   # run a race, show E[position]
pytest                                    # 11 tests, incl. seed determinism
python benchmarks/bench_engine.py        # throughput receipts
```

## What's modeled (0.1)

Per-lap pace via a pluggable `PaceModel` (stub: car offset + compound offset +
linear/quadratic tyre degradation + fuel burn + per-driver noise), pit stops
with stochastic stationary time, and a per-lap safety-car hazard with sampled
duration, slowed laps, and discounted pit loss.

## Known simplifications (tracked)

No traffic/dirty air, no SC field bunching, no overtaking difficulty (order is
cumulative time), no two-compound rule, no DNFs. These land alongside the
trained model in 0.2+, where `StubPaceModel` is replaced by posterior samples
from the hierarchical Bayesian model (ADR-0003) behind the same `PaceModel`
protocol.

## Benchmark (reference machine, 57 laps × 20 cars)

| rollouts | time | races/s |
|---:|---:|---:|
| 1 | 2.4 ms | 410 |
| 1,000 | 0.066 s | ~15,000 |
| 10,000 | 0.60 s | ~16,600 |

Phase 0.1 exit criterion (1,000 rollouts < 1 s): **PASS**.
