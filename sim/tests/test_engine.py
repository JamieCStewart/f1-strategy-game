"""Behavioral tests for the engine's racing logic."""
import dataclasses

import numpy as np
import pytest

from pitwall_sim.config import CarConfig, CircuitConfig, Compound, RaceConfig
from pitwall_sim.defaults import default_config, default_strategies
from pitwall_sim.engine import run_race
from pitwall_sim.strategy import PitStop, StrategyPlan, validate_strategies


def _quiet_config(laps: int = 20, sc_prob: float = 0.0) -> RaceConfig:
    """Two-car, zero-noise, controllable-SC config for exact assertions."""
    circuit = CircuitConfig(
        name="Test Ring",
        laps=laps,
        base_lap_time_s=90.0,
        pit_loss_s=20.0,
        pit_stop_sd_s=0.0,
        sc_prob_per_lap=sc_prob,
        sc_min_laps=3,
        sc_max_laps=3,
    )
    cars = (
        CarConfig(0, "A", 0.0, 0.0),
        CarConfig(1, "B", 0.5, 0.0),
    )
    return RaceConfig(circuit=circuit, cars=cars)


def test_shapes():
    config = default_config(laps=25)
    r = run_race(config, default_strategies(config), n_rollouts=16, seed=0)
    assert r.lap_times.shape == (16, 25, config.n_cars)
    assert r.sc_active.shape == (16, 25)
    assert r.final_time.shape == (16, config.n_cars)
    assert r.final_positions().shape == (16, config.n_cars)


def test_final_time_is_sum_of_lap_times():
    config = default_config(laps=25)
    r = run_race(config, default_strategies(config), n_rollouts=4, seed=3)
    np.testing.assert_allclose(r.lap_times.sum(axis=1), r.final_time)


def test_pit_stop_costs_time_and_resets_degradation():
    config = _quiet_config(laps=12)
    strategies = (
        StrategyPlan(Compound.SOFT, (PitStop(6, Compound.SOFT),)),
        StrategyPlan(Compound.SOFT, ()),
    )
    r = run_race(config, strategies, seed=0)
    pit_lap = r.lap_times[0, 5, 0]       # lap 6 includes the stop
    normal_lap = r.lap_times[0, 5, 1]
    assert pit_lap > normal_lap + config.circuit.pit_loss_s * 0.9
    # Fresh tyres after the stop: car 0's lap 7 beats car 0's lap 5,
    # by more than the fuel-burn gain alone.
    fuel_gain = 2 * config.fuel.burn_kg_per_lap * config.fuel.effect_s_per_kg
    assert r.lap_times[0, 6, 0] < r.lap_times[0, 4, 0] - fuel_gain / 2


def test_degradation_and_fuel_burn_shape_the_stint():
    config = _quiet_config(laps=15)
    strategies = (
        StrategyPlan(Compound.HARD, ()),
        StrategyPlan(Compound.HARD, ()),
    )
    r = run_race(config, strategies, seed=0)
    stint = r.lap_times[0, :, 0]
    deltas = np.diff(stint)
    # Stub model: hard-tyre deg (~0.03 s/lap) < fuel gain (~0.058 s/lap)
    # early on, so lap times fall, and the quadratic term bends the curve up.
    assert deltas[0] < 0
    assert deltas[-1] > deltas[0]


def test_safety_car_slows_the_field():
    config = _quiet_config(laps=10, sc_prob=1.0)  # SC triggers on lap 1
    strategies = (
        StrategyPlan(Compound.MEDIUM, ()),
        StrategyPlan(Compound.MEDIUM, ()),
    )
    r = run_race(config, strategies, seed=0)
    assert r.sc_active[0, :3].all()
    sc_floor = config.circuit.base_lap_time_s * 1.2
    assert (r.lap_times[0, :3, :] > sc_floor).all()


def test_faster_car_usually_wins():
    config = default_config(laps=40)
    r = run_race(config, default_strategies(config), n_rollouts=200, seed=11)
    mean_pos = r.final_positions().mean(axis=0)
    assert mean_pos[0] < mean_pos[-1]  # fastest car beats slowest on average


def test_strategy_validation_rejects_bad_plans():
    config = _quiet_config(laps=10)
    with pytest.raises(ValueError):
        validate_strategies((StrategyPlan(Compound.SOFT, ()),), config)  # wrong count
    bad_order = (
        StrategyPlan(
            Compound.SOFT,
            (PitStop(7, Compound.HARD), PitStop(3, Compound.HARD)),
        ),
        StrategyPlan(Compound.SOFT, ()),
    )
    with pytest.raises(ValueError):
        validate_strategies(bad_order, config)
    out_of_range = (
        StrategyPlan(Compound.SOFT, (PitStop(10, Compound.HARD),)),
        StrategyPlan(Compound.SOFT, ()),
    )
    with pytest.raises(ValueError):
        validate_strategies(out_of_range, config)


def test_config_is_immutable():
    config = _quiet_config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.circuit.laps = 99  # type: ignore[misc]
