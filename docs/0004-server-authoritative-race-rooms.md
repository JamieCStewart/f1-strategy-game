# ADR-0004: Server-authoritative race rooms with pause-on-decision timing

**Status:** Accepted · 2026-06-10

## Context

Three timing models were considered: real-time accelerated (tension, but punishes thinking), turn-based (thoughtful, but drains race drama), and a hybrid that runs accelerated and pauses when the player faces a decision. Separately: simulation could run in the browser (cheap, but trivially cheatable and exposes the model) or on the server. Multiplayer is planned for v2 and must not require a rewrite.

## Decision

The simulation runs server-side inside a **race room** owned by a worker process. The room advances on a wall-clock tick cadence and enters a `decision_pending` state — clock stopped — whenever one of the player's cars reaches a decision window. Clients are renderers connected by WebSocket; the only client→server game input is a `Decision` message validated against the room's offered choices. Rooms are registered in Redis; state ticks fan out via Redis pub/sub.

## Consequences

- Hybrid pacing: races stay tense (~20–30 min) but every call gets thinking time.
- No client-side cheating; the trained model never ships to the browser.
- Multiplayer later = multiple humans per room + pause semantics for concurrent decisions (a contained change), because the room abstraction exists from day one.
- The room/worker split defines the scaling unit: capacity is "rooms per node."
- Cost: server compute per active race; acceptable because one room is bounded CPU and v1 is solo.
