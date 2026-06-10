"""A default fictional 20-car grid and sensible strategies (ADR-0005).

Lives outside the engine core: this is demo/test content, not simulation
logic. Names are invented; pace offsets sketch a plausible competitive spread.
"""
from __future__ import annotations

from .config import CarConfig, CircuitConfig, Compound, RaceConfig
from .strategy import PitStop, StrategyPlan

_TEAMS = [
    ("Vortex", 0.00),
    ("Borealis", 0.15),
    ("Solara", 0.40),
    ("Kestrel", 0.65),
    ("Meridian", 0.95),
    ("Halcyon", 1.20),
    ("Tempest", 1.50),
    ("Auriga", 1.85),
    ("Polaris", 2.15),
    ("Corsair", 2.50),
]


def default_circuit(laps: int = 57) -> CircuitConfig:
    return CircuitConfig(
        name="Autodromo Levante",
        laps=laps,
        base_lap_time_s=88.0,
        pit_loss_s=20.5,
    )


def default_config(laps: int = 57) -> RaceConfig:
    cars = []
    for t, (team, offset) in enumerate(_TEAMS):
        for d in (1, 2):
            cars.append(
                CarConfig(
                    car_id=t * 2 + d - 1,
                    name=f"{team} {d}",
                    pace_offset_s=offset + (d - 1) * 0.12,
                    noise_sd_s=0.15 + 0.006 * t,
                )
            )
    return RaceConfig(circuit=default_circuit(laps), cars=tuple(cars))


def default_strategies(config: RaceConfig) -> tuple[StrategyPlan, ...]:
    """Alternate one-stop (M->H) and two-stop (S->M->M) down the grid."""
    L = config.circuit.laps
    one_stop = StrategyPlan(
        Compound.MEDIUM, (PitStop(round(L * 0.50), Compound.HARD),)
    )
    two_stop = StrategyPlan(
        Compound.SOFT,
        (
            PitStop(round(L * 0.30), Compound.MEDIUM),
            PitStop(round(L * 0.65), Compound.MEDIUM),
        ),
    )
    return tuple(
        one_stop if i % 2 == 0 else two_stop for i in range(config.n_cars)
    )
