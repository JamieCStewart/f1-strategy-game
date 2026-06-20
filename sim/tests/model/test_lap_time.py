"""Tests for the PyMC model definition.

Uses prior predictive sampling (no MCMC) so CI runs in seconds.
Full posterior sampling is done offline; see the training script.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pymc", reason="pymc not installed; skipping model tests")

from pitwall_sim.model.features import encode_circuits, to_model_arrays
from pitwall_sim.model.lap_time import build_model, extract_draws
from pitwall_sim.model.artifact import PosteriorDraws

from .conftest import synthetic_laps


@pytest.fixture(scope="module")
def small_arrays():
    df = synthetic_laps(n_laps=60, circuits=["Barcelona", "Monza"], seed=1)
    df_enc, mapping = encode_circuits(df)
    arrays = to_model_arrays(df_enc, mapping)
    return arrays, mapping


def test_model_builds(small_arrays):
    arrays, mapping = small_arrays
    model = build_model(arrays, mapping)
    assert model is not None


def test_model_prior_predictive(small_arrays):
    import pymc as pm

    arrays, mapping = small_arrays
    model = build_model(arrays, mapping)
    with model:
        prior = pm.sample_prior_predictive(samples=10, random_seed=0)
    assert "lap_time_obs" in prior.prior_predictive


def test_prior_predictive_lap_times_are_plausible(small_arrays):
    import pymc as pm

    arrays, mapping = small_arrays
    model = build_model(arrays, mapping)
    with model:
        prior = pm.sample_prior_predictive(samples=50, random_seed=42)
    # Median of prior predictive should be in a sane range for F1
    pp = prior.prior_predictive["lap_time_obs"].values.flatten()
    assert np.median(pp) > 70.0
    assert np.median(pp) < 120.0


def test_extract_draws_shapes():
    """extract_draws should work on a mock InferenceData-like object."""
    import types

    n_chains, n_draws_per_chain, n_circuits = 2, 100, 2
    circuit_map = {"Barcelona": 0, "Monza": 1}

    # Build a minimal mock of az.InferenceData.posterior
    post = types.SimpleNamespace(
        **{
            var: types.SimpleNamespace(
                values=np.random.default_rng(0).normal(
                    size=(n_chains, n_draws_per_chain, *shape)
                )
            )
            for var, shape in [
                ("circuit_base", (n_circuits,)),
                ("compound_offset", (3,)),
                ("deg_lin", (3,)),
                ("deg_quad", (3,)),
                ("fuel_effect", ()),
                ("sigma_lap", (3,)),
            ]
        }
    )
    idata = types.SimpleNamespace(posterior=post)

    draws = extract_draws(idata, "Barcelona", circuit_map, n_draws=50)
    assert isinstance(draws, PosteriorDraws)
    assert draws.circuit_base.shape == (50,)
    assert draws.compound_offset.shape == (50, 3)
    assert draws.noise_sd.shape == (50, 3)
