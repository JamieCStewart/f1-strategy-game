import { useState, useEffect, useRef, useCallback } from "react";
import { AppState, RacingState, DecisionState, FinishedState } from "./types";
import { RaceHeader } from "./components/RaceHeader";
import { TimingTower } from "./components/TimingTower";
import { DecisionModal } from "./components/DecisionModal";
import { FinishedScreen } from "./components/FinishedScreen";

export default function App() {
  const [appState, setAppState] = useState<AppState>({ phase: "lobby" });
  const wsRef = useRef<WebSocket | null>(null);

  const startRace = useCallback(async () => {
    const res = await fetch("/api/races", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        tick_interval: 4.0,
        seed: Math.floor(Math.random() * 10000),
      }),
    });
    const { race_id } = (await res.json()) as { race_id: string };

    const wsProto = window.location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${wsProto}://${window.location.host}/ws/${race_id}`);
    wsRef.current = ws;

    ws.onmessage = (event: MessageEvent) => {
      const msg = JSON.parse(event.data as string) as Record<string, unknown>;

      setAppState((prev) => {
        switch (msg.type) {
          case "race_started":
            return {
              phase: "racing",
              lap: 0,
              totalLaps: msg.total_laps as number,
              cars: [],
              scActive: false,
              playerCarIndices: msg.player_car_indices as number[],
            };

          case "lap_complete": {
            if (prev.phase !== "racing" && prev.phase !== "decision") return prev;
            const base = prev as RacingState | DecisionState;
            return {
              ...base,
              phase: "racing",
              lap: msg.lap as number,
              cars: msg.cars as RacingState["cars"],
              scActive: msg.sc_active as boolean,
            };
          }

          case "decision_prompt":
            if (prev.phase !== "racing" && prev.phase !== "decision") return prev;
            return {
              ...(prev as RacingState | DecisionState),
              phase: "decision",
              prompt: msg as unknown as DecisionState["prompt"],
            };

          case "race_finished":
            return {
              phase: "finished",
              classification: msg.classification as FinishedState["classification"],
              playerCarIndices:
                prev.phase !== "lobby" && prev.phase !== "finished"
                  ? (prev as RacingState).playerCarIndices
                  : [],
            };

          default:
            return prev;
        }
      });
    };

    ws.onerror = () => console.error("WebSocket error");
  }, []);

  const sendDecision = useCallback((carIdx: number, action: string) => {
    wsRef.current?.send(
      JSON.stringify({ type: "player_decision", car_idx: carIdx, action })
    );
  }, []);

  useEffect(() => {
    return () => wsRef.current?.close();
  }, []);

  if (appState.phase === "lobby") {
    return (
      <div style={{
        display: "flex", alignItems: "center", justifyContent: "center",
        height: "100vh", background: "#0d0d0d", flexDirection: "column", gap: 0,
      }}>
        <div style={{ fontSize: 11, letterSpacing: 4, color: "#555", marginBottom: 10 }}>
          F1 RACE STRATEGY
        </div>
        <div style={{ fontSize: 28, letterSpacing: 6, color: "#e0e0e0", marginBottom: 48 }}>
          PIT WALL
        </div>
        <button
          onClick={startRace}
          style={{
            background: "#e8002d", color: "#fff", border: "none",
            padding: "12px 36px", fontSize: 13, letterSpacing: 3, cursor: "pointer",
          }}
          onMouseEnter={(e) => { (e.currentTarget as HTMLButtonElement).style.background = "#c0001f"; }}
          onMouseLeave={(e) => { (e.currentTarget as HTMLButtonElement).style.background = "#e8002d"; }}
        >
          START RACE
        </button>
        <div style={{ marginTop: 24, fontSize: 11, color: "#333", letterSpacing: 1 }}>
          Autodromo Levante · 57 laps · 20 cars · You control Vortex 1 &amp; 2
        </div>
      </div>
    );
  }

  if (appState.phase === "finished") {
    return (
      <FinishedScreen
        state={appState}
        onRestart={() => {
          wsRef.current?.close();
          wsRef.current = null;
          setAppState({ phase: "lobby" });
        }}
      />
    );
  }

  const live = appState as RacingState | DecisionState;

  return (
    <div style={{ background: "#0d0d0d", minHeight: "100vh" }}>
      <RaceHeader
        lap={live.lap}
        totalLaps={live.totalLaps}
        scActive={live.scActive}
        phase={live.phase}
      />
      <TimingTower cars={live.cars} playerCarIndices={live.playerCarIndices} />
      {live.phase === "decision" && (
        <DecisionModal
          prompt={(live as DecisionState).prompt}
          onDecide={sendDecision}
        />
      )}
    </div>
  );
}

