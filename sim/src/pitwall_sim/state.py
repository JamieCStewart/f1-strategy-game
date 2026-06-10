"""Batched race state: struct-of-arrays with a rollout dimension (ADR-0007).

Every per-car quantity is an array of shape (n_rollouts, n_cars). The live
race is simply n_rollouts == 1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import numpy as _np

from .config import RaceConfig


@dataclass
class RaceState:
    lap: int                      # next lap to be run, 1-indexed
    cumulative_time: np.ndarray   # (R, C) float64, seconds
    tyre_age: np.ndarray          # (R, C) int64, laps on current set
    compound: np.ndarray          # (R, C) int64, Compound codes
    fuel_kg: np.ndarray           # (R, C) float64
    sc_laps_remaining: np.ndarray  # (R,) int64, 0 = green flag

    @property
    def n_rollouts(self) -> int:
        return self.cumulative_time.shape[0]

    @property
    def sc_active(self) -> np.ndarray:  # (R,) bool
        return self.sc_laps_remaining > 0

    @classmethod
    def initial(
        cls,
        config: RaceConfig,
        start_compounds: "np.ndarray",
        n_rollouts: int,
    ) -> "RaceState":
        shape = (n_rollouts, config.n_cars)
        start_compounds = np.asarray(start_compounds, dtype=np.int64)
        return cls(
            lap=1,
            cumulative_time=np.zeros(shape, dtype=np.float64),
            tyre_age=np.zeros(shape, dtype=np.int64),
            compound=np.broadcast_to(start_compounds, shape).copy(),
            fuel_kg=np.full(shape, config.fuel.start_kg, dtype=np.float64),
            sc_laps_remaining=np.zeros(n_rollouts, dtype=np.int64),
        )

    def tiled(self, n_rollouts: int) -> "RaceState":
        """Copy of this single-rollout state broadcast to a rollout batch."""
        assert self.n_rollouts == 1, "tile from a live (R=1) state"
        rep = lambda a: _np.repeat(a, n_rollouts, axis=0).copy()
        return RaceState(
            lap=self.lap,
            cumulative_time=rep(self.cumulative_time),
            tyre_age=rep(self.tyre_age),
            compound=rep(self.compound),
            fuel_kg=rep(self.fuel_kg),
            sc_laps_remaining=_np.repeat(self.sc_laps_remaining, n_rollouts).copy(),
        )
