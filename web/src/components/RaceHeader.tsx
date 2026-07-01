interface Props {
  lap: number;
  totalLaps: number;
  scActive: boolean;
  phase: string;
}

export function RaceHeader({ lap, totalLaps, scActive, phase }: Props) {
  return (
    <div style={{
      background: "#111",
      borderBottom: "1px solid #222",
      padding: "10px 16px",
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center",
    }}>
      <div style={{ fontSize: 14, letterSpacing: 3, color: "#e0e0e0", fontWeight: "bold" }}>
        PIT WALL
      </div>

      <div style={{ display: "flex", gap: 24, alignItems: "center", fontSize: 12 }}>
        {scActive && (
          <span style={{ color: "#ff8c00", letterSpacing: 2, animation: "pulse 1.2s ease-in-out infinite" }}>
            SC DEPLOYED
          </span>
        )}
        {phase === "decision" && (
          <span style={{ color: "#e8002d", letterSpacing: 2, animation: "pulse 0.8s ease-in-out infinite" }}>
            DECISION REQUIRED
          </span>
        )}
        {lap > 0 && (
          <span style={{ color: "#aaa", letterSpacing: 2 }}>
            LAP {lap} / {totalLaps}
          </span>
        )}
      </div>
    </div>
  );
}
