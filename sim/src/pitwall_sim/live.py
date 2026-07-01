"""LiveRace and MC continuation — the real-time face of the sim engine.

The live race is a batch of size 1 (ADR-0007); `continue_race` broadcasts it
to a rollout batch via `state.tiled()`. Both use `advance_one_lap` as their
shared physics kernel so the live race and the AI planner are provably running
the same simulation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import Compound, RaceConfig
from .engine import RaceResult, _pit_table, advance_one_lap
from .pace import PaceModel, StubPaceModel
from .state import RaceState
from .strategy import StrategyPlan, validate_strategies


@dataclass
class LapRecord:
    """Outcome of one completed lap in the live race."""

    lap: int                     # 1-indexed lap number just completed
    lap_times: np.ndarray        # (C,) total duration including any pit time
    sc_active: bool              # was safety car active this lap?
    pit_this_lap: np.ndarray     # (C,) bool — stop taken at end of this lap


@dataclass
class RolloutResult:
    """MC continuation results: shape (R, L_remaining, C)."""

    config: RaceConfig
    seed: int
    lap_times: np.ndarray    # (R, L_remaining, C)
    sc_active: np.ndarray    # (R, L_remaining) bool
    final_time: np.ndarray   # (R, C) cumulative time at the end of the race

    def final_positions(self) -> np.ndarray:
        """Finishing position per rollout per car, shape (R, C), 1-indexed."""
        order = np.argsort(self.final_time, axis=1)
        ranks = np.empty_like(order)
        rows = np.arange(order.shape[0])[:, None]
        ranks[rows, order] = np.arange(1, order.shape[1] + 1)[None, :]
        return ranks


class LiveRace:
    """Single-rollout live race (the game's authoritative race state).

    Advances lap-by-lap; call `step()` once per game tick. Use
    `continue_race` or `state.tiled()` to spin off MC rollouts for the
    AI planner and prediction service.
    """

    def __init__(
        self,
        config: RaceConfig,
        strategies: tuple[StrategyPlan, ...],
        *,
        seed: int = 0,
        pace_model: PaceModel | None = None,
    ) -> None:
        validate_strategies(strategies, config)
        self._config = config
        self._strategies = strategies
        self._model = pace_model if pace_model is not None else StubPaceModel(config)
        self._rng = np.random.default_rng(np.random.SeedSequence(seed))
        start = np.array([int(p.start_compound) for p in strategies], dtype=np.int64)
        self._state = RaceState.initial(config, start, n_rollouts=1)
        self._pit_table = _pit_table(config, strategies)
        self._records: list[LapRecord] = []

    @property
    def state(self) -> RaceState:
        return self._state

    @property
    def lap(self) -> int:
        return self._state.lap

    @property
    def laps_remaining(self) -> int:
        return self._config.circuit.laps - self._state.lap + 1

    @property
    def finished(self) -> bool:
        return self._state.lap > self._config.circuit.laps

    @property
    def records(self) -> list[LapRecord]:
        return list(self._records)

    def queue_stop(self, car_idx: int, compound: Compound) -> None:
        """Queue a pit stop for car_idx at the end of the current (upcoming) lap.

        Must be called before step(); has no effect if called after.
        """
        self._pit_table[self._state.lap, car_idx] = int(compound)

    def step(self) -> LapRecord:
        """Advance the live race by one lap and return the lap record."""
        if self.finished:
            raise RuntimeError("Race is already finished")
        lap = self._state.lap
        pit_compound = self._pit_table[lap]
        times, sc = advance_one_lap(
            self._config, self._model, self._state, self._rng, pit_compound
        )
        record = LapRecord(
            lap=lap,
            lap_times=times[0],           # squeeze rollout dim
            sc_active=bool(sc[0]),
            pit_this_lap=pit_compound >= 0,
        )
        self._records.append(record)
        return record


def plans_to_pit_table(
    config: RaceConfig,
    strategies: tuple[StrategyPlan, ...],
) -> np.ndarray:
    """Return the (L+1, C) pit-compound table for a set of strategies.

    Entry [l, c] is the compound fitted at the end of lap l, or -1 for no
    stop. Exported here for consumers that need the table without running a
    full race (e.g. the AI planner pre-computing stop windows).
    """
    return _pit_table(config, strategies)


def continue_race(
    config: RaceConfig,
    state: RaceState,
    strategies: tuple[StrategyPlan, ...],
    *,
    n_rollouts: int,
    seed: int = 0,
    pace_model: PaceModel | None = None,
) -> RolloutResult:
    """Continue a race from `state` to the finish through `n_rollouts` MC runs.

    `state` is typically the current live-race state (n_rollouts=1); it is
    broadcast to the rollout batch via `state.tiled()`. The caller's state
    object is not mutated.

    Used by:
    - AI planner: evaluates candidate strategies from the current lap.
    - Prediction service: produces ensemble fan charts (ADR-0008).
    """
    validate_strategies(strategies, config)
    model = pace_model if pace_model is not None else StubPaceModel(config)
    rng = np.random.default_rng(np.random.SeedSequence(seed))

    batch_state = state.tiled(n_rollouts) if state.n_rollouts == 1 else state
    pit_table = _pit_table(config, strategies)

    laps_left = config.circuit.laps - state.lap + 1
    R, C = batch_state.cumulative_time.shape
    lap_times = np.empty((R, laps_left, C), dtype=np.float64)
    sc_log = np.empty((R, laps_left), dtype=bool)

    for i, lap in enumerate(range(state.lap, config.circuit.laps + 1)):
        times, sc = advance_one_lap(
            config, model, batch_state, rng, pit_table[lap]
        )
        lap_times[:, i, :] = times
        sc_log[:, i] = sc

    return RolloutResult(
        config=config,
        seed=seed,
        lap_times=lap_times,
        sc_active=sc_log,
        final_time=batch_state.cumulative_time.copy(),
    )
