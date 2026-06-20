"""Tests for calibration and backtest metrics."""
from __future__ import annotations

import numpy as np
import pytest

from pitwall_sim.model.artifact import PosteriorDraws
from pitwall_sim.model.features import encode_circuits
from pitwall_sim.model.validate import (
    BacktestMetrics,
    compute_metrics,
    posterior_predictive_quantiles,
)

from .conftest import synthetic_draws, synthetic_laps


def test_posterior_predictive_shape():
    draws = synthetic_draws(n_draws=50)
    df = synthetic_laps(n_laps=30, circuits=["Barcelona"])
    _, mapping = encode_circuits(df)
    from pitwall_sim.model.features import to_model_arrays
    arrays = to_model_arrays(df, mapping)
    pp = posterior_predictive_quantiles(
        draws, arrays, np.zeros(1), n_samples=40, rng=np.random.default_rng(0)
    )
    assert pp.shape == (30, 40)


def test_compute_metrics_returns_metrics():
    draws = synthetic_draws(circuit_id="Barcelona", n_draws=100)
    df = synthetic_laps(n_laps=80, circuits=["Barcelona"])
    _, mapping = encode_circuits(df)
    metrics = compute_metrics(draws, df, mapping, rng=np.random.default_rng(0))
    assert isinstance(metrics, BacktestMetrics)
    assert metrics.n_laps == 80
    assert metrics.rmse_s > 0
    assert 0.0 <= metrics.coverage_80 <= 1.0
    assert 0.0 <= metrics.coverage_95 <= 1.0


def test_well_calibrated_model_passes_gate():
    """A model trained on its own data should be well calibrated."""
    rng = np.random.default_rng(99)
    # Build draws whose parameters match what the synthetic data was generated with
    n = 500
    draws = PosteriorDraws(
        circuit_id="Barcelona",
        circuit_base=np.full(n, 91.0),
        compound_offset=np.tile([0.0, 0.7, 1.2], (n, 1)),
        deg_lin=np.tile([0.08, 0.05, 0.03], (n, 1)),
        deg_quad=np.zeros((n, 3)),
        fuel_effect=np.full(n, 0.032),
        noise_sd=np.full((n, 3), 0.3),
    )
    df = synthetic_laps(n_laps=200, circuits=["Barcelona"], seed=7)
    _, mapping = encode_circuits(df)
    metrics = compute_metrics(draws, df, mapping, rng=rng)
    assert metrics.passes_gate(), (
        f"Expected calibrated model to pass gate; got {metrics}"
    )


def test_bad_model_fails_gate():
    rng = np.random.default_rng(0)
    n = 200
    # Draws with very wrong base time → huge RMSE
    draws = PosteriorDraws(
        circuit_id="Barcelona",
        circuit_base=np.full(n, 200.0),  # 200 s instead of ~91 s
        compound_offset=np.zeros((n, 3)),
        deg_lin=np.zeros((n, 3)),
        deg_quad=np.zeros((n, 3)),
        fuel_effect=np.zeros(n),
        noise_sd=np.full((n, 3), 0.1),
    )
    df = synthetic_laps(n_laps=50, circuits=["Barcelona"])
    _, mapping = encode_circuits(df)
    metrics = compute_metrics(draws, df, mapping, rng=rng)
    assert not metrics.passes_gate()


def test_coverage_80_increases_with_wider_noise():
    rng = np.random.default_rng(5)
    df = synthetic_laps(n_laps=100, circuits=["Barcelona"])
    _, mapping = encode_circuits(df)
    base_kwargs = dict(
        circuit_id="Barcelona",
        circuit_base=np.full(200, 91.0),
        compound_offset=np.tile([0.0, 0.7, 1.2], (200, 1)),
        deg_lin=np.tile([0.08, 0.05, 0.03], (200, 1)),
        deg_quad=np.zeros((200, 3)),
        fuel_effect=np.full(200, 0.032),
    )
    narrow = PosteriorDraws(**base_kwargs, noise_sd=np.full((200, 3), 0.01))
    wide = PosteriorDraws(**base_kwargs, noise_sd=np.full((200, 3), 5.0))

    m_narrow = compute_metrics(narrow, df, mapping, rng=np.random.default_rng(6))
    m_wide = compute_metrics(wide, df, mapping, rng=np.random.default_rng(7))
    assert m_wide.coverage_80 > m_narrow.coverage_80
