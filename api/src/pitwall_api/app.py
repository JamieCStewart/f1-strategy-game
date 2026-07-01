"""FastAPI application: REST + WebSocket endpoints for Pit Wall.

Phase 0.3: single-process, in-memory rooms. No Redis, no Postgres.
One WebSocket connection per race; the tick loop starts on first connect.
"""
from __future__ import annotations

import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .room import RaceRoom, build_default_room

app = FastAPI(title="Pit Wall API", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_rooms: dict[str, RaceRoom] = {}
_tasks: dict[str, asyncio.Task] = {}  # type: ignore[type-arg]


class CreateRaceRequest(BaseModel):
    tick_interval: float = 4.0
    seed: int = 42


@app.post("/api/races")
async def create_race(req: CreateRaceRequest) -> dict:
    import uuid
    race_id = str(uuid.uuid4())[:8]
    room = build_default_room(race_id, tick_interval=req.tick_interval, seed=req.seed)
    _rooms[race_id] = room
    return {"race_id": race_id}


@app.get("/api/races/{race_id}")
async def get_race(race_id: str) -> dict:
    room = _rooms.get(race_id)
    if room is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Race not found")
    return {"race_id": race_id, "phase": room.phase}


@app.websocket("/ws/{race_id}")
async def ws_race(websocket: WebSocket, race_id: str) -> None:
    await websocket.accept()

    room = _rooms.get(race_id)
    if room is None:
        await websocket.send_json({"type": "error", "message": "Race not found"})
        await websocket.close()
        return

    async def send(msg: dict) -> None:
        try:
            await websocket.send_json(msg)
        except Exception:
            pass

    room.subscribe(send)

    # Start the tick loop on first connection (idempotent).
    if race_id not in _tasks or _tasks[race_id].done():
        _tasks[race_id] = asyncio.create_task(room.run())

    try:
        while True:
            data = await websocket.receive_json()
            if data.get("type") == "player_decision":
                room.receive_decision(int(data["car_idx"]), str(data["action"]))
    except WebSocketDisconnect:
        room.unsubscribe(send)
    except Exception:
        room.unsubscribe(send)


def main() -> None:
    import uvicorn
    uvicorn.run("pitwall_api.app:app", host="0.0.0.0", port=8000, reload=True)
