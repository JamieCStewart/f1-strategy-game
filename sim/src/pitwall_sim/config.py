"""Static race configuration: circuit, cars, tyre compounds.

Pure data. No I/O, no randomness (ADR-0002).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class Compound(IntEnum):
    SOFT = 0
    MEDIUM = 1
    HARD = 2


@dataclass(frozen=True)
class CompoundParams:
    """Stub degradation parameters; replaced by the trained model in 0.2."""

    offset_s: float        # pace offset vs. reference compound (s/lap, fresh)
    deg_linear: float      # s/lap per lap of tyre age
    deg_quadratic: float   # s/lap per lap-of-age squared


DEFAULT_COMPOUNDS: dict[Compound, CompoundParams] = {
    Compound.SOFT: CompoundParams(-0.7, 0.080, 0.0015),
    Compound.MEDIUM: CompoundParams(0.0, 0.050, 0.0008),
    Compound.HARD: CompoundParams(0.5, 0.030, 0.0004),
}


@dataclass(frozen=True)
class CircuitConfig:
    name: str
    laps: int
    base_lap_time_s: float        # reference dry pace, fresh medium, zero fuel
    pit_loss_s: float             # total pit-lane time loss at racing speed
    pit_stop_mean_s: float = 2.5  # stationary time, mean
    pit_stop_sd_s: float = 0.4    # stationary time, spread
    sc_prob_per_lap: float = 0.012
    sc_min_laps: int = 2
    sc_max_laps: int = 5
    sc_pace_factor: float = 1.40  # lap time multiplier under safety car
    sc_pit_loss_factor: float = 0.55  # pit loss is cheaper under SC


@dataclass(frozen=True)
class CarConfig:
    car_id: int
    name: str
    pace_offset_s: float   # car+driver pace vs. the reference (s/lap)
    noise_sd_s: float      # lap-to-lap consistency


@dataclass(frozen=True)
class FuelConfig:
    start_kg: float = 105.0
    burn_kg_per_lap: float = 1.8
    effect_s_per_kg: float = 0.032


@dataclass(frozen=True)
class RaceConfig:
    circuit: CircuitConfig
    cars: tuple[CarConfig, ...]
    fuel: FuelConfig = field(default_factory=FuelConfig)
    compounds: dict[Compound, CompoundParams] = field(
        default_factory=lambda: dict(DEFAULT_COMPOUNDS)
    )

    @property
    def n_cars(self) -> int:
        return len(self.cars)
