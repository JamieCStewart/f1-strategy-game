"""Benchmark: rollout throughput (ADR-0007).

Run:  python benchmarks/bench_engine.py
Target (Phase 0.1 exit criterion): a 1,000-rollout batch of a full
57-lap, 20-car race in under 1 second.
"""
from __future__ import annotations

import time

from pitwall_sim.defaults import default_config, default_strategies
from pitwall_sim.engine import run_race


def bench(n_rollouts: int, repeats: int = 5) -> float:
    config = default_config(laps=57)
    strategies = default_strategies(config)
    run_race(config, strategies, n_rollouts=n_rollouts, seed=0)  # warm-up
    best = float("inf")
    for i in range(repeats):
        t0 = time.perf_counter()
        run_race(config, strategies, n_rollouts=n_rollouts, seed=i)
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    print(f"{'rollouts':>9} {'best (s)':>10} {'races/s':>12} {'us/race':>9}")
    for n in (1, 10, 100, 1_000, 10_000):
        secs = bench(n)
        print(f"{n:>9} {secs:>10.4f} {n / secs:>12,.0f} {1e6 * secs / n:>9.1f}")
    one_k = bench(1_000)
    status = "PASS" if one_k < 1.0 else "FAIL"
    print(f"\nPhase 0.1 exit criterion (1,000 rollouts < 1 s): "
          f"{one_k:.3f} s -> {status}")


if __name__ == "__main__":
    main()
