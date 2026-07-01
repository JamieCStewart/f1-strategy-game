# Testing Phase 0.3 — Pit Wall

## Prerequisites

```bash
# Python deps (one-time)
cd sim && pip install -e . && cd ..
cd api && pip install -e . && cd ..

# Node deps (one-time)
cd web && npm install && cd ..
```

## Run (two terminals)

**Terminal 1 — API server:**
```bash
cd /path/to/f1-strategy-game
uvicorn pitwall_api.app:app --reload --port 8000
```
Expected: `Uvicorn running on http://127.0.0.1:8000`

**Terminal 2 — Web dev server:**
```bash
cd /path/to/f1-strategy-game/web
npm run dev
```
Expected: `Local: http://localhost:5173/`

Then open **http://localhost:5173** in a browser.

---

## What to expect

### Lobby screen
- Dark background, "PIT WALL" heading, red **START RACE** button
- Subtitle: *Autodromo Levante · 57 laps · 20 cars · You control Vortex 1 & 2*

### Click START RACE
- Race creates and WebSocket connects
- "FORMATION LAP" placeholder appears briefly
- Timing tower populates after lap 1 (≈4 seconds)
- **Blue left border** = your cars (Vortex 1, Vortex 2)
- Columns: POS · DRIVER · INTERVAL · GAP · TYR · AGE · LAST LAP

### Decision modals
Two decision windows per race for Vortex 2 (two-stopper), one for Vortex 1:

| Car | Window opens | Window closes |
|-----|-------------|---------------|
| Vortex 2 | Lap 14 | Lap 25 |
| Vortex 1 | Lap 25 | Lap 36 |
| Vortex 2 | Lap 34 | Lap 45 |

When a window opens the race **pauses**, a modal appears:
- **Stay Out** — skip this lap, re-prompts next lap
- **Pit — Soft / Medium / Hard** — boxes the car end of this lap

**Tip:** Pitting on Soft late in the race (lap 40+) is fast but risky on deg. Medium is a safe bet.

### Safety car
Header shows `SC DEPLOYED` in orange (pulsing) when active. Pit loss is cheaper under SC (~11s instead of ~20s) — worth considering if you're near a window.

### Finished screen
Full 20-car classification. Your cars highlighted in blue. "RACE AGAIN" resets to lobby.

---

## Speed up for quick testing

Change `tick_interval` in the POST body (sent by the frontend) to make laps faster. Edit `web/src/App.tsx` line with `tick_interval: 4.0` → `tick_interval: 0.5` and save. Vite hot-reloads instantly.

---

## Quick API smoke test (no browser)

```bash
# Create race
RACE=$(curl -s -X POST http://localhost:8000/api/races \
  -H "Content-Type: application/json" \
  -d '{"tick_interval": 0, "seed": 42}' | python3 -c "import sys,json; print(json.load(sys.stdin)['race_id'])")

echo "Race ID: $RACE"
curl -s http://localhost:8000/api/races/$RACE | python3 -m json.tool
```

---

## Plugin Claude needs to test frontend apps

To let me drive and screenshot the UI myself, the project needs:

**`chromium-cli`** — the tool the `/run` skill expects. Install via:
```bash
npm install -g @anthropic-ai/chromium-cli
# or check: https://github.com/anthropics/chromium-cli
```

If that package isn't available, the fallback is `playwright`:
```bash
pip install playwright && python3 -m playwright install chromium
```

With either installed, running `/run` in a future session will let me open
http://localhost:5173, click START RACE, wait for laps, screenshot the
timing tower, interact with decision modals, and report what I actually see —
rather than asking you to do it.
