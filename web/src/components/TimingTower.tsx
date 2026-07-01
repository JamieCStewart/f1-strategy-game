import { CarState } from "../types";

const COMPOUND_COLOR: Record<string, string> = {
  S: "#e8002d",
  M: "#ffd700",
  H: "#e8e8e8",
};

function fmtLapTime(s: number): string {
  if (s <= 0) return "--:--.---";
  const m = Math.floor(s / 60);
  const rem = s - m * 60;
  return `${m}:${rem.toFixed(3).padStart(6, "0")}`;
}

function fmtGap(gap: number): string {
  if (gap === 0) return "LEADER";
  return `+${gap.toFixed(3)}`;
}

function fmtInterval(interval: number, pos: number): string {
  if (pos === 1) return "";
  return `+${interval.toFixed(3)}`;
}

interface Props {
  cars: CarState[];
  playerCarIndices: number[];
}

export function TimingTower({ cars, playerCarIndices }: Props) {
  const playerSet = new Set(playerCarIndices);
  const sorted = [...cars].sort((a, b) => a.position - b.position);

  if (sorted.length === 0) {
    return (
      <div style={{ padding: "40px 16px", color: "#555", fontSize: 13, letterSpacing: 2, textAlign: "center" }}>
        FORMATION LAP
      </div>
    );
  }

  const COL = "32px 180px 90px 100px 28px 40px 100px";

  return (
    <div style={{ padding: "0 16px 16px" }}>
      <div style={{
        display: "grid",
        gridTemplateColumns: COL,
        columnGap: 8,
        color: "#555",
        fontSize: 10,
        padding: "6px 4px",
        borderBottom: "1px solid #222",
        letterSpacing: 2,
      }}>
        <span>POS</span>
        <span>DRIVER</span>
        <span>INTERVAL</span>
        <span>GAP</span>
        <span>TYR</span>
        <span>AGE</span>
        <span>LAST LAP</span>
      </div>

      {sorted.map((car) => {
        const isPlayer = playerSet.has(car.car_idx);
        const cmpColor = COMPOUND_COLOR[car.compound] ?? "#aaa";
        const posColor = car.position === 1 ? "#ffd700" : car.position <= 3 ? "#c0c0c0" : "#888";

        return (
          <div
            key={car.car_idx}
            style={{
              display: "grid",
              gridTemplateColumns: COL,
              columnGap: 8,
              padding: "5px 4px",
              background: isPlayer
                ? "rgba(79, 195, 247, 0.06)"
                : car.position % 2 === 0 ? "#0f0f0f" : "#0d0d0d",
              borderBottom: "1px solid #1a1a1a",
              borderLeft: isPlayer ? "2px solid #4fc3f7" : "2px solid transparent",
              alignItems: "center",
              fontSize: 13,
            }}
          >
            <span style={{ color: posColor, fontWeight: "bold" }}>
              {car.position}
            </span>
            <span style={{ color: isPlayer ? "#4fc3f7" : "#ccc", overflow: "hidden", whiteSpace: "nowrap" }}>
              {car.car_name}
            </span>
            <span style={{ color: "#999" }}>
              {fmtInterval(car.interval, car.position)}
            </span>
            <span style={{ color: "#aaa" }}>
              {fmtGap(car.gap_to_leader)}
            </span>
            <span style={{ color: cmpColor, fontWeight: "bold" }}>
              {car.compound}
            </span>
            <span style={{ color: "#666" }}>
              {car.tyre_age}
            </span>
            <span style={{ color: car.pitted_this_lap ? "#4fc3f7" : "#ccc" }}>
              {fmtLapTime(car.last_lap_time)}
              {car.pitted_this_lap && (
                <span style={{ color: "#4fc3f7", marginLeft: 6, fontSize: 10 }}>PIT</span>
              )}
            </span>
          </div>
        );
      })}
    </div>
  );
}
