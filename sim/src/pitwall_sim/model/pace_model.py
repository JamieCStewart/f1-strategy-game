"""BayesianPaceModel: drop-in replacement for StubPaceModel.

Implements the PaceModel protocol (pace.py) by indexing pre-drawn posterior
samples. No PyMC at runtime — only numpy.

Usage:
    draws = PosteriorDraws.load("artifacts/barcelona.npz")
    car_offsets = np.array([c.pace_offset_s for c in config.cars])
    model = BayesianPaceModel(draws, car_offsets)
    result = run_race(config, strategies, n_rollouts=1000, pace_model=model)
"""
from __future__ import annotations

import numpy as np

from ..state import RaceState
from .artifact import PosteriorDraws


class BayesianPaceModel:
    """Samples lap times from pre-drawn posterior parameters.

    Each rollout gets one posterior draw, drawn uniformly at random. This
    correctly propagates parameter uncertainty across the rollout batch:
    rollouts that happened to draw a "slow circuit day" posterior sample will
    consistently produce slower laps throughout the race.
    """

    def __init__(
        self,
        draws: PosteriorDraws,
        car_offsets: np.ndarray,
    ) -> None:
        """
        Args:
            draws: pre-sampled posterior for the target circuit.
            car_offsets: (n_cars,) pace offsets in seconds, one per car.
                Phase 0.2 sources these from RaceConfig (not yet learned);
                phase 0.3+ will replace with trained car-season posteriors.
        """
        self.draws = draws
        self.car_offsets = np.asarray(car_offsets, dtype=np.float64)

    def sample_lap_times(
        self, state: RaceState, rng: np.random.Generator
    ) -> np.ndarray:
        """Return green-flag lap times in seconds, shape (n_rollouts, n_cars)."""
        R, C = state.cumulative_time.shape
        draw_idx = rng.integers(0, self.draws.n_draws, size=R)

        comp = state.compound           # (R, C) int
        age = state.tyre_age.astype(np.float64)  # (R, C)
        rows = np.arange(R)

        base = self.draws.circuit_base[draw_idx][:, None]        # (R, 1)
        c_off = self.draws.compound_offset[draw_idx]             # (R, 3)
        dl = self.draws.deg_lin[draw_idx]                        # (R, 3)
        dq = self.draws.deg_quad[draw_idx]                       # (R, 3)
        fuel_eff = self.draws.fuel_effect[draw_idx][:, None]     # (R, 1)
        sigma = self.draws.noise_sd[draw_idx]                    # (R, 3)

        mu = (
            base
            + self.car_offsets[None, :]                          # (1, C)
            + c_off[rows[:, None], comp]                         # (R, C)
            + dl[rows[:, None], comp] * age                      # (R, C)
            + dq[rows[:, None], comp] * age ** 2                 # (R, C)
            + fuel_eff * state.fuel_kg                           # (R, C)
        )
        sigma_rc = sigma[rows[:, None], comp]                    # (R, C)
        return mu + rng.normal(0.0, 1.0, size=(R, C)) * sigma_rc
