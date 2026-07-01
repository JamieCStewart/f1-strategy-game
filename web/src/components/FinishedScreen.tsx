import { FinishedState } from "../types";

function fmtGap(gap: number): string {
  if (gap === 0) return "WINNER";
  return `+${gap.toFixed(3)}s`;
}

interface Props {
  state: FinishedState;
  onRestart: () => void;
}

export function FinishedScreen({ state, onRestart }: Props) {
  const playerSet = new Set(state.playerCarIndices);

  return (
    <div style={{
      minHeight: "100vh",
      background: "#0d0d0d",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      paddingTop: 60,
    }}>
      <div style={{ fontSize: 11, letterSpacing: 4, color: "#555", marginBottom: 8 }}>
        RACE CLASSIFICATION
      </div>
      <div style={{ fontSize: 22, letterSpacing: 3, color: "#e0e0e0", marginBottom: 40 }}>
        RACE FINISHED
      </div>

      <div style={{ width: "100%", maxWidth: 500, padding: "0 16px" }}>
        {state.classification.map((entry) => {
          const isPlayer = playerSet.has(entry.car_idx);
          const posColor =
            entry.position === 1 ? "#ffd700" :
            entry.position === 2 ? "#c0c0c0" :
            entry.position === 3 ? "#cd7f32" : "#666";

          return (
            <div
              key={entry.car_idx}
              style={{
                display: "grid",
                gridTemplateColumns: "36px 1fr 120px",
                columnGap: 12,
                padding: "7px 12px",
                background: isPlayer ? "rgba(79, 195, 247, 0.06)" : "transparent",
                borderLeft: isPlayer ? "2px solid #4fc3f7" : "2px solid transparent",
                borderBottom: "1px solid #1a1a1a",
                alignItems: "center",
                fontSize: 13,
              }}
            >
              <span style={{ color: posColor, fontWeight: "bold" }}>{entry.position}</span>
              <span style={{ color: isPlayer ? "#4fc3f7" : "#ccc" }}>{entry.car_name}</span>
              <span style={{ color: entry.position === 1 ? "#ffd700" : "#777", textAlign: "right" }}>
                {fmtGap(entry.gap)}
              </span>
            </div>
          );
        })}
      </div>

      <button
        onClick={onRestart}
        style={{
          marginTop: 48,
          background: "transparent",
          border: "1px solid #333",
          color: "#888",
          padding: "10px 28px",
          fontSize: 12,
          letterSpacing: 2,
          cursor: "pointer",
        }}
        onMouseEnter={(e) => { (e.currentTarget as HTMLButtonElement).style.borderColor = "#666"; }}
        onMouseLeave={(e) => { (e.currentTarget as HTMLButtonElement).style.borderColor = "#333"; }}
      >
        RACE AGAIN
      </button>
    </div>
  );
}
