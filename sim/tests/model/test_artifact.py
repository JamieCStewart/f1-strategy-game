"""Round-trip save/load tests for PosteriorDraws."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from pitwall_sim.model.artifact import PosteriorDraws

from .conftest import synthetic_draws


def test_save_load_roundtrip(tmp_path):
    draws = synthetic_draws(circuit_id="Barcelona", n_draws=20)
    path = tmp_path / "test.npz"
    draws.save(path)
    loaded = PosteriorDraws.load(path)

    assert loaded.circuit_id == "Barcelona"
    np.testing.assert_array_equal(loaded.circuit_base, draws.circuit_base)
    np.testing.assert_array_equal(loaded.compound_offset, draws.compound_offset)
    np.testing.assert_array_equal(loaded.deg_lin, draws.deg_lin)
    np.testing.assert_array_equal(loaded.deg_quad, draws.deg_quad)
    np.testing.assert_array_equal(loaded.fuel_effect, draws.fuel_effect)
    np.testing.assert_array_equal(loaded.noise_sd, draws.noise_sd)


def test_n_draws_property():
    draws = synthetic_draws(n_draws=37)
    assert draws.n_draws == 37


def test_load_accepts_string_path(tmp_path):
    draws = synthetic_draws(n_draws=5)
    path = str(tmp_path / "draws.npz")
    draws.save(path)
    loaded = PosteriorDraws.load(path)
    assert loaded.n_draws == 5
