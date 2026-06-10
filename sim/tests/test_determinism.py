"""ADR-0002: a race is a deterministic function of (config, strategies, seed)."""
import numpy as np

from pitwall_sim.defaults import default_config, default_strategies
from pitwall_sim.engine import run_race


def _run(seed: int, rollouts: int = 8):
    config = default_config(laps=30)
    return run_race(
        config, default_strategies(config), n_rollouts=rollouts, seed=seed
    )


def test_same_seed_is_byte_identical():
    a, b = _run(seed=42), _run(seed=42)
    np.testing.assert_array_equal(a.lap_times, b.lap_times)
    np.testing.assert_array_equal(a.final_time, b.final_time)
    np.testing.assert_array_equal(a.sc_active, b.sc_active)


def test_different_seeds_differ():
    a, b = _run(seed=1), _run(seed=2)
    assert not np.array_equal(a.final_time, b.final_time)


def test_rollouts_within_batch_differ():
    r = _run(seed=7, rollouts=4)
    assert not np.array_equal(r.lap_times[0], r.lap_times[1])
