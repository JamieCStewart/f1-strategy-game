"""Backtest and calibration metrics for the lap-time model.

The CI gate calls `check_calibration_gate` against a committed fixture
dataset; the same functions are used in notebooks for exploratory analysis.

Key metrics:
  - coverage_p: fraction of laps where actual falls within the [p/2, 1-p/2]
                posterior predictive interval (target: ~p for a calibrated model)
  - rmse_s:     root-mean-squared error of the posterior predictive median vs actual
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .artifact import PosteriorDraws
from .features import to_model_arrays


@dataclass
class BacktestMetrics:
    n_laps: int
    rmse_s: float
    coverage_80: float   # fraction within 10th–90th posterior predictive interval
    coverage_95: float   # fraction within 2.5th–97.5th

    def passes_gate(
        self,
        min_coverage_80: float = 0.75,
        max_rmse_s: float = 1.5,
    ) -> bool:
        """Return True if this model artifact meets the CI promotion threshold."""
        return self.coverage_80 >= min_coverage_80 and self.rmse_s <= max_rmse_s


def posterior_predictive_quantiles(
    draws: PosteriorDraws,
    arrays: dict[str, np.ndarray],
    car_offsets: np.ndarray,
    *,
    n_samples: int = 2000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate posterior predictive samples for a set of laps.

    Returns an array of shape (n_laps, n_samples) containing simulated lap
    times drawn from the posterior predictive distribution.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    n_laps = len(arrays["lap_time_s"])
    compound = arrays["compound"]      # (n_laps,)
    tyre_age = arrays["tyre_age"]      # (n_laps,)
    fuel_kg = arrays["fuel_kg"]        # (n_laps,)
    circuit_idx = arrays["circuit_idx"]  # (n_laps,) — single circuit expected

    # Draw n_samples parameter sets
    draw_idx = rng.integers(0, draws.n_draws, size=n_samples)

    base = draws.circuit_base[draw_idx]            # (S,)
    c_off = draws.compound_offset[draw_idx]        # (S, 3)
    dl = draws.deg_lin[draw_idx]                   # (S, 3)
    dq = draws.deg_quad[draw_idx]                  # (S, 3)
    fuel_eff = draws.fuel_effect[draw_idx]         # (S,)
    sigma = draws.noise_sd[draw_idx]               # (S, 3)

    s_idx = np.arange(n_samples)

    # mu: (n_laps, n_samples)
    mu = (
        base[None, :]                              # (1, S)
        + car_offsets[None, :]                     # broadcast ignored — no car dim here
        + c_off[s_idx[None, :], compound[:, None]] # (N, S)
        + dl[s_idx[None, :], compound[:, None]] * tyre_age[:, None]
        + dq[s_idx[None, :], compound[:, None]] * tyre_age[:, None] ** 2
        + fuel_eff[None, :] * fuel_kg[:, None]
    )
    sigma_ns = sigma[s_idx[None, :], compound[:, None]]  # (N, S)
    return mu + rng.normal(0.0, 1.0, size=(n_laps, n_samples)) * sigma_ns


def compute_metrics(
    draws: PosteriorDraws,
    test_df: pd.DataFrame,
    circuit_map: dict[str, int],
    car_offsets: np.ndarray | None = None,
    *,
    n_samples: int = 2000,
    rng: np.random.Generator | None = None,
) -> BacktestMetrics:
    """Compute BacktestMetrics for a held-out DataFrame."""
    arrays = to_model_arrays(test_df, circuit_map)
    if car_offsets is None:
        # No car offsets known at backtest time: use zeros
        car_offsets = np.zeros(1, dtype=np.float64)

    pp = posterior_predictive_quantiles(
        draws, arrays, car_offsets, n_samples=n_samples, rng=rng
    )
    y = arrays["lap_time_s"]

    median_pred = np.median(pp, axis=1)
    rmse = float(np.sqrt(np.mean((y - median_pred) ** 2)))

    lo_80 = np.percentile(pp, 10, axis=1)
    hi_80 = np.percentile(pp, 90, axis=1)
    lo_95 = np.percentile(pp, 2.5, axis=1)
    hi_95 = np.percentile(pp, 97.5, axis=1)

    cov_80 = float(np.mean((y >= lo_80) & (y <= hi_80)))
    cov_95 = float(np.mean((y >= lo_95) & (y <= hi_95)))

    return BacktestMetrics(
        n_laps=len(y),
        rmse_s=rmse,
        coverage_80=cov_80,
        coverage_95=cov_95,
    )
