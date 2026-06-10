"""Strategy plans: the decision schema for Phase 0.1.

A plan is the starting compound plus an ordered list of pit stops. Stops are
1-indexed by lap and executed at the END of that lap. In 0.3 this grows into
the live Decision message; for now plans are fixed before the race and applied
uniformly across all rollouts in a batch (each candidate strategy gets its own
batch — see ADR-0008).
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import Compound, RaceConfig


@dataclass(frozen=True)
class PitStop:
    lap: int
    compound: Compound


@dataclass(frozen=True)
class StrategyPlan:
    start_compound: Compound
    stops: tuple[PitStop, ...]


def validate_strategies(
    strategies: tuple[StrategyPlan, ...], config: RaceConfig
) -> None:
    if len(strategies) != config.n_cars:
        raise ValueError(
            f"need one strategy per car: got {len(strategies)} "
            f"for {config.n_cars} cars"
        )
    for car, plan in zip(config.cars, strategies):
        laps = [s.lap for s in plan.stops]
        if laps != sorted(laps) or len(set(laps)) != len(laps):
            raise ValueError(f"{car.name}: pit laps must be strictly increasing")
        if laps and (laps[0] < 1 or laps[-1] >= config.circuit.laps):
            raise ValueError(
                f"{car.name}: pit laps must lie in [1, {config.circuit.laps - 1}]"
            )
