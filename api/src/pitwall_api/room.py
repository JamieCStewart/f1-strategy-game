"""RaceRoom: in-memory race state machine with an async tick loop.

Each room owns one LiveRace. The tick loop advances the race lap-by-lap,
pausing at player decision windows to send a DecisionPrompt and wait for
the client's PlayerDecision message before continuing.

Phase 0.3: in-memory only (no Redis, no Postgres). AI cars follow fixed
default strategies; player cars decide dynamically via WebSocket messages.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Awaitable, Callable

import numpy as np

from pitwall_sim.config import Compound, RaceConfig
from pitwall_sim.defaults import default_config, default_strategies
from pitwall_sim.live import LiveRace, LapRecord
from pitwall_sim.strategy import StrategyPlan

_COMPOUND_NAMES: dict[int, str] = {0: "S", 1: "M", 2: "H"}
_COMPOUND_BY_LETTER: dict[str, Compound] = {
    "S": Compound.SOFT,
    "M": Compound.MEDIUM,
    "H": Compound.HARD,
}

PLAYER_CAR_INDICES = [0, 1]


@dataclass
class DecisionWindow:
    car_idx: int
    open_lap: int
    close_lap: int
    resolved: bool = False
    last_prompted_lap: int = -1


class RaceRoom:
    def __init__(
        self,
        race_id: str,
        config: RaceConfig,
        strategies: tuple[StrategyPlan, ...],
        decision_windows: list[DecisionWindow],
        tick_interval: float = 4.0,
        seed: int = 0,
    ) -> None:
        self.race_id = race_id
        self.phase = "running"

        self._config = config
        self._tick_interval = tick_interval
        self._player_cars: set[int] = set(PLAYER_CAR_INDICES)
        self._windows = decision_windows

        self._live = LiveRace(config, strategies, seed=seed)

        self._pending_car: int | None = None
        self._decision_event: asyncio.Event = asyncio.Event()
        self._decision_action: str | None = None

        self._callbacks: list[Callable[[dict], Awaitable[None]]] = []

    def subscribe(self, cb: Callable[[dict], Awaitable[None]]) -> None:
        self._callbacks.append(cb)

    def unsubscribe(self, cb: Callable[[dict], Awaitable[None]]) -> None:
        self._callbacks = [c for c in self._callbacks if c is not cb]

    async def _broadcast(self, msg: dict) -> None:
        for cb in list(self._callbacks):
            try:
                await cb(msg)
            except Exception:
                pass

    def _compound_name(self, code: int) -> str:
        return _COMPOUND_NAMES.get(code, "?")

    def _car_states(self, record: LapRecord) -> list[dict]:
        state = self._live.state
        times = state.cumulative_time[0]
        order = np.argsort(times)
        leader_time = float(times[order[0]])

        result = []
        for pos, car_idx in enumerate(order, 1):
            t = float(times[car_idx])
            prev_t = float(times[order[pos - 2]]) if pos > 1 else t
            result.append({
                "car_idx": int(car_idx),
                "car_name": self._config.cars[car_idx].name,
                "position": pos,
                "gap_to_leader": round(t - leader_time, 3),
                "interval": round(t - prev_t, 3) if pos > 1 else 0.0,
                "compound": self._compound_name(int(state.compound[0, car_idx])),
                "tyre_age": int(state.tyre_age[0, car_idx]),
                "last_lap_time": round(float(record.lap_times[car_idx]), 3),
                "cumulative_time": round(t, 3),
                "pitted_this_lap": bool(record.pit_this_lap[car_idx]),
                "is_player": int(car_idx) in self._player_cars,
            })
        return result

    def _next_window(self, lap: int) -> DecisionWindow | None:
        for w in self._windows:
            if w.resolved:
                continue
            if lap > w.close_lap:
                w.resolved = True
                continue
            if w.open_lap <= lap and w.last_prompted_lap != lap:
                return w
        return None

    def _build_prompt(self, w: DecisionWindow) -> dict:
        state = self._live.state
        car_idx = w.car_idx
        code = int(state.compound[0, car_idx])
        age = int(state.tyre_age[0, car_idx])
        current = self._compound_name(code)
        lap = self._live.lap

        options: list[dict] = [
            {
                "action": "stay_out",
                "label": "Stay Out",
                "description": f"Continue on {current} ({age} laps). Re-prompts next lap.",
            }
        ]
        pit_loss = self._config.circuit.pit_loss_s
        for letter, name in [("S", "Soft"), ("M", "Medium"), ("H", "Hard")]:
            options.append({
                "action": f"pit_{letter}",
                "label": f"Pit — {name}",
                "description": f"Box this lap, fit new {name.lower()}s. Pit loss ~{pit_loss:.0f}s.",
            })

        return {
            "type": "decision_prompt",
            "lap": lap,
            "car_idx": car_idx,
            "car_name": self._config.cars[car_idx].name,
            "tyre_age": age,
            "current_compound": current,
            "options": options,
            "window_closes": w.close_lap,
        }

    async def run(self) -> None:
        """Main race loop. Runs as an asyncio task, started on first WS connect."""
        car_names = [c.name for c in self._config.cars]
        await self._broadcast({
            "type": "race_started",
            "race_id": self.race_id,
            "total_laps": self._config.circuit.laps,
            "player_car_indices": sorted(self._player_cars),
            "car_names": car_names,
        })

        while not self._live.finished:
            upcoming = self._live.lap

            # Drain all decision windows that open this lap before stepping.
            while True:
                w = self._next_window(upcoming)
                if w is None:
                    break

                self.phase = "decision_pending"
                self._pending_car = w.car_idx
                w.last_prompted_lap = upcoming

                await self._broadcast(self._build_prompt(w))
                await self._decision_event.wait()
                self._decision_event.clear()

                action = self._decision_action
                self._decision_action = None
                self._pending_car = None
                self.phase = "running"

                if action and action.startswith("pit_"):
                    compound = _COMPOUND_BY_LETTER[action[4:]]
                    self._live.queue_stop(w.car_idx, compound)
                    w.resolved = True
                # else: stay_out — leave window unresolved to re-prompt next lap

            record = self._live.step()

            await self._broadcast({
                "type": "lap_complete",
                "lap": record.lap,
                "total_laps": self._config.circuit.laps,
                "sc_active": record.sc_active,
                "cars": self._car_states(record),
            })

            if not self._live.finished:
                await asyncio.sleep(self._tick_interval)

        state = self._live.state
        times = state.cumulative_time[0]
        order = np.argsort(times)
        leader_time = float(times[order[0]])

        classification = [
            {
                "position": pos,
                "car_idx": int(idx),
                "car_name": self._config.cars[idx].name,
                "total_time": round(float(times[idx]), 3),
                "gap": round(float(times[idx]) - leader_time, 3),
                "is_player": int(idx) in self._player_cars,
            }
            for pos, idx in enumerate(order, 1)
        ]

        await self._broadcast({"type": "race_finished", "classification": classification})
        self.phase = "finished"

    def receive_decision(self, car_idx: int, action: str) -> bool:
        """Called by the WS handler when a player sends a decision message."""
        if self.phase != "decision_pending" or self._pending_car != car_idx:
            return False
        self._decision_action = action
        self._decision_event.set()
        return True


def _player_strategies(all_strats: tuple[StrategyPlan, ...]) -> tuple[StrategyPlan, ...]:
    """Override player car strategies with no pre-planned stops."""
    strats = list(all_strats)
    for idx in PLAYER_CAR_INDICES:
        strats[idx] = StrategyPlan(start_compound=Compound.MEDIUM, stops=())
    return tuple(strats)


def _build_windows(
    config: RaceConfig,
    all_strats: tuple[StrategyPlan, ...],
) -> list[DecisionWindow]:
    """Derive player decision windows from the default strategy stop laps."""
    windows: list[DecisionWindow] = []
    L = config.circuit.laps
    for car_idx in PLAYER_CAR_INDICES:
        for stop in all_strats[car_idx].stops:
            windows.append(DecisionWindow(
                car_idx=car_idx,
                open_lap=max(2, stop.lap - 3),
                close_lap=min(L - 2, stop.lap + 8),
            ))
    return windows


def build_default_room(
    race_id: str,
    tick_interval: float = 4.0,
    seed: int = 0,
) -> RaceRoom:
    config = default_config()
    all_strats = default_strategies(config)
    return RaceRoom(
        race_id=race_id,
        config=config,
        strategies=_player_strategies(all_strats),
        decision_windows=_build_windows(config, all_strats),
        tick_interval=tick_interval,
        seed=seed,
    )
