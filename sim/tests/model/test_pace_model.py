"""Tests for BayesianPaceModel — no PyMC, no FastF1."""
from __future__ import annotations

import numpy as np
import pytest

from pitwall_sim.config import RaceConfig, CircuitConfig, CarConfig, Compound
from pitwall_sim.state import RaceState
from pitwall_sim.model.pace_model import BayesianPaceModel

from .conftest import synthetic_draws


def _config(n_cars: int = 4) -> RaceConfig:
    circuit = CircuitConfig(
        name="Test", laps=50, base_lap_time_s=91.0, pit_loss_s=22.0
    )
    cars = tuple(
        CarConfig(i, f"Car{i}", pace_offset_s=float(i) * 0.1, noise_sd_s=0.3)
        for i in range(n_cars)
    )
    return RaceConfig(circuit=circuit, cars=cars)


def _state(config: RaceConfig, n_rollouts: int = 8) -> RaceState:
    start = np.array([int(Compound.MEDIUM)] * config.n_cars, dtype=np.int64)
    return RaceState.initial(config, start, n_rollouts)


def test_output_shape():
    config = _config(n_cars=4)
    state = _state(config, n_rollouts=8)
    draws = synthetic_draws(circuit_id="Barcelona", n_draws=50)
    car_offsets = np.array([c.pace_offset_s for c in config.cars])
    model = BayesianPaceModel(draws, car_offsets)
    rng = np.random.default_rng(0)
    times = model.sample_lap_times(state, rng)
    assert times.shape == (8, 4)


def test_output_is_float64():
    config = _config()
    state = _state(config)
    draws = synthetic_draws(n_draws=50)
    model = BayesianPaceModel(draws, np.zeros(config.n_cars))
    times = model.sample_lap_times(state, np.random.default_rng(1))
    assert times.dtype == np.float64


def test_lap_times_are_plausible():
    config = _config()
    state = _state(config, n_rollouts=200)
    draws = synthetic_draws(circuit_id="Barcelona", n_draws=100)
    model = BayesianPaceModel(draws, np.zeros(config.n_cars))
    times = model.sample_lap_times(state, np.random.default_rng(2))
    assert times.min() > 70.0
    assert times.max() < 130.0


def test_different_rollouts_can_differ():
    """Each rollout should draw a different posterior sample."""
    config = _config()
    state = _state(config, n_rollouts=100)
    draws = synthetic_draws(n_draws=100)
    model = BayesianPaceModel(draws, np.zeros(config.n_cars))
    times = model.sample_lap_times(state, np.random.default_rng(3))
    # At least some rollouts should differ
    assert not np.all(times == times[0])


def test_faster_car_offset_produces_lower_times():
    config = _config(n_cars=2)
    state = _state(config, n_rollouts=500)
    draws = synthetic_draws(n_draws=200)
    # car 0 is 1 second faster than car 1
    car_offsets = np.array([-0.5, 0.5])
    model = BayesianPaceModel(draws, car_offsets)
    times = model.sample_lap_times(state, np.random.default_rng(4))
    assert times[:, 0].mean() < times[:, 1].mean()


def test_satisfies_pace_model_protocol():
    """Structural check: runtime confirm it matches the PaceModel protocol."""
    from pitwall_sim.pace import PaceModel
    import inspect

    model = BayesianPaceModel(synthetic_draws(n_draws=10), np.zeros(4))
    # Protocol compliance: must have sample_lap_times(state, rng) -> ndarray
    assert hasattr(model, "sample_lap_times")
    sig = inspect.signature(model.sample_lap_times)
    assert list(sig.parameters.keys()) == ["state", "rng"]
