"""The sim <-> model contract (ARCHITECTURE.md section 5.4).

The engine only ever calls `sample_lap_times`. The trained hierarchical
Bayesian model (Phase 0.2) will implement this same protocol by indexing into
pre-drawn posterior samples; StubPaceModel is a transparent parametric
stand-in so the engine, tests, and benchmarks exist before any data does.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from .config import Compound, RaceConfig
from .state import RaceState


class PaceModel(Protocol):
    def sample_lap_times(
        self, state: RaceState, rng: np.random.Generator
    ) -> np.ndarray:
        """Return green-flag lap times in seconds, shape (n_rollouts, n_cars)."""
        ...


class StubPaceModel:
    """base + car offset + compound offset + tyre degradation + fuel + noise."""

    def __init__(self, config: RaceConfig):
        self.base = config.circuit.base_lap_time_s
        self.car_offset = np.array(
            [c.pace_offset_s for c in config.cars], dtype=np.float64
        )
        self.noise_sd = np.array(
            [c.noise_sd_s for c in config.cars], dtype=np.float64
        )
        comps = [config.compounds[Compound(i)] for i in sorted(Compound)]
        self.comp_offset = np.array([p.offset_s for p in comps])
        self.deg_lin = np.array([p.deg_linear for p in comps])
        self.deg_quad = np.array([p.deg_quadratic for p in comps])
        self.fuel_effect = config.fuel.effect_s_per_kg

    def sample_lap_times(
        self, state: RaceState, rng: np.random.Generator
    ) -> np.ndarray:
        comp = state.compound
        age = state.tyre_age.astype(np.float64)
        times = (
            self.base
            + self.car_offset[None, :]
            + self.comp_offset[comp]
            + self.deg_lin[comp] * age
            + self.deg_quad[comp] * age * age
            + self.fuel_effect * state.fuel_kg
        )
        times += rng.normal(0.0, 1.0, size=comp.shape) * self.noise_sd[None, :]
        return times
