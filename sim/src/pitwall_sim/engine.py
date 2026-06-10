"""The race engine: a vectorized lap-by-lap loop over a batch of rollouts.

Pure function of (config, strategies, seed) -> RaceResult (ADR-0002): no I/O,
no clocks, all randomness from the injected seed. All per-car math is array
math over shape (n_rollouts, n_cars) (ADR-0007).

Known 0.1 simplifications (tracked for later phases): no traffic/dirty-air
effect, no field bunching under safety car, no overtaking difficulty (order is
purely cumulative time), no two-compound rule enforcement, no incidents/DNFs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import RaceConfig
from .pace import PaceModel, StubPaceModel
from .state import RaceState
from .strategy import StrategyPlan, validate_strategies


@dataclass
class RaceResult:
    config: RaceConfig
    strategies: tuple[StrategyPlan, ...]
    seed: int
    lap_times: np.ndarray     # (R, L, C) total lap duration incl. pit time
    pit_laps: np.ndarray      # (L, C) bool, stop executed at end of that lap
    sc_active: np.ndarray     # (R, L) bool
    final_time: np.ndarray    # (R, C) seconds

    def final_positions(self) -> np.ndarray:
        """Finishing position per rollout per car, shape (R, C), 1-indexed."""
        order = np.argsort(self.final_time, axis=1)
        ranks = np.empty_like(order)
        rows = np.arange(order.shape[0])[:, None]
        ranks[rows, order] = np.arange(1, order.shape[1] + 1)[None, :]
        return ranks

    def classification(self, rollout: int = 0) -> list[tuple[int, float]]:
        """[(car_index, gap_to_leader_s), ...] in finishing order."""
        times = self.final_time[rollout]
        order = np.argsort(times)
        return [(int(c), float(times[c] - times[order[0]])) for c in order]


def _pit_table(
    config: RaceConfig, strategies: tuple[StrategyPlan, ...]
) -> np.ndarray:
    """(L+1, C) int: compound fitted at the end of lap l, or -1 for no stop."""
    table = np.full((config.circuit.laps + 1, config.n_cars), -1, dtype=np.int64)
    for car_idx, plan in enumerate(strategies):
        for stop in plan.stops:
            table[stop.lap, car_idx] = int(stop.compound)
    return table


def advance_one_lap(
    config: RaceConfig,
    model: PaceModel,
    state: RaceState,
    rng: np.random.Generator,
    pit_compound: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance every rollout by one lap, mutating `state` in place.

    pit_compound: (n_cars,) int, the compound fitted at the END of this lap,
    or -1 to stay out. Returns (lap_times (R, C), sc_active (R,)).
    The single physics path shared by run_race, LiveRace, and continue_race.
    """
    circuit = config.circuit
    R, C = state.cumulative_time.shape

    # Safety car process (per rollout)
    green = state.sc_laps_remaining == 0
    trigger = green & (rng.random(R) < circuit.sc_prob_per_lap)
    durations = rng.integers(circuit.sc_min_laps, circuit.sc_max_laps + 1, size=R)
    state.sc_laps_remaining = np.where(trigger, durations, state.sc_laps_remaining)
    sc = state.sc_active

    # Driving time
    green_times = model.sample_lap_times(state, rng)
    sc_times = circuit.base_lap_time_s * circuit.sc_pace_factor + rng.normal(
        0.0, 0.3, size=(R, C)
    )
    times = np.where(sc[:, None], sc_times, green_times)

    # Pit stops at the end of this lap
    pitting = pit_compound >= 0
    if pitting.any():
        loss = circuit.pit_loss_s * np.where(
            sc[:, None], circuit.sc_pit_loss_factor, 1.0
        )
        stationary = np.clip(
            rng.normal(circuit.pit_stop_mean_s, circuit.pit_stop_sd_s, size=(R, C)),
            circuit.pit_stop_mean_s - 3 * circuit.pit_stop_sd_s,
            None,
        )
        times = times + pitting[None, :] * (loss + stationary)

    # Advance state
    state.cumulative_time += times
    state.tyre_age += 1
    if pitting.any():
        state.tyre_age = np.where(pitting[None, :], 0, state.tyre_age)
        state.compound = np.where(
            pitting[None, :], np.broadcast_to(pit_compound, (R, C)), state.compound
        )
    state.fuel_kg = np.maximum(state.fuel_kg - config.fuel.burn_kg_per_lap, 0.0)
    state.sc_laps_remaining = np.maximum(state.sc_laps_remaining - 1, 0)
    state.lap += 1
    return times, sc


def run_race(
    config: RaceConfig,
    strategies: tuple[StrategyPlan, ...],
    *,
    n_rollouts: int = 1,
    seed: int = 0,
    pace_model: PaceModel | None = None,
) -> RaceResult:
    validate_strategies(strategies, config)
    model = pace_model if pace_model is not None else StubPaceModel(config)
    rng = np.random.default_rng(np.random.SeedSequence(seed))

    R, C, L = n_rollouts, config.n_cars, config.circuit.laps
    start = np.array([int(p.start_compound) for p in strategies], dtype=np.int64)
    state = RaceState.initial(config, start, R)
    pit_table = _pit_table(config, strategies)

    lap_times = np.empty((R, L, C), dtype=np.float64)
    sc_log = np.empty((R, L), dtype=bool)
    for lap in range(1, L + 1):
        times, sc = advance_one_lap(config, model, state, rng, pit_table[lap])
        lap_times[:, lap - 1, :] = times
        sc_log[:, lap - 1] = sc

    return RaceResult(
        config=config,
        strategies=strategies,
        seed=seed,
        lap_times=lap_times,
        pit_laps=pit_table[1:] >= 0,
        sc_active=sc_log,
        final_time=state.cumulative_time.copy(),
    )
