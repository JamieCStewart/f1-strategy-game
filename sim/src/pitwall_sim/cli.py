"""CLI race runner: `python -m pitwall_sim.cli --seed 42 --rollouts 1000`.

A thin presentation layer; all simulation stays in the pure engine.
"""
from __future__ import annotations

import argparse

import numpy as np

from .defaults import default_config, default_strategies
from .engine import run_race


def _fmt_gap(seconds: float) -> str:
    if seconds == 0.0:
        return "WINNER"
    return f"+{seconds:7.3f}s"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Run a simulated race.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--laps", type=int, default=57)
    p.add_argument("--rollouts", type=int, default=1)
    args = p.parse_args(argv)

    config = default_config(laps=args.laps)
    strategies = default_strategies(config)
    result = run_race(
        config, strategies, n_rollouts=args.rollouts, seed=args.seed
    )

    sc_laps = np.flatnonzero(result.sc_active[0]) + 1
    print(f"\n  {config.circuit.name} — {config.circuit.laps} laps, "
          f"seed {args.seed}, {args.rollouts} rollout(s)")
    print(f"  Safety car (rollout 0): "
          f"{', '.join(map(str, sc_laps)) if sc_laps.size else 'none'}\n")

    mean_pos = result.final_positions().mean(axis=0)
    header = f"  {'P':>2}  {'Car':<14} {'Gap':>10}"
    if args.rollouts > 1:
        header += f"  {'E[pos]':>7}"
    print(header)
    for pos, (car_idx, gap) in enumerate(result.classification(0), start=1):
        line = f"  {pos:>2}  {config.cars[car_idx].name:<14} {_fmt_gap(gap):>10}"
        if args.rollouts > 1:
            line += f"  {mean_pos[car_idx]:>7.2f}"
        print(line)
    print()


if __name__ == "__main__":
    main()
