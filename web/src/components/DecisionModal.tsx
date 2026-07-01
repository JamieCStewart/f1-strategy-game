import { DecisionPrompt, DecisionOption } from "../types";

const COMPOUND_COLOR: Record<string, string> = {
  S: "#e8002d",
  M: "#ffd700",
  H: "#e8e8e8",
};

function compoundLetter(action: string): string | null {
  if (action.startsWith("pit_")) return action[4];
  return null;
}

interface OptionButtonProps {
  opt: DecisionOption;
  onClick: () => void;
}

function OptionButton({ opt, onClick }: OptionButtonProps) {
  const letter = compoundLetter(opt.action);
  const color = letter ? (COMPOUND_COLOR[letter] ?? "#aaa") : "#555";
  const isStayOut = opt.action === "stay_out";

  return (
    <button
      onClick={onClick}
      style={{
        background: isStayOut ? "#1a1a1a" : "#0d0d0d",
        border: `1px solid ${color}`,
        color,
        padding: "11px 14px",
        cursor: "pointer",
        textAlign: "left",
        transition: "background 0.1s",
        width: "100%",
      }}
      onMouseEnter={(e) => {
        (e.currentTarget as HTMLButtonElement).style.background = isStayOut ? "#222" : "#141414";
      }}
      onMouseLeave={(e) => {
        (e.currentTarget as HTMLButtonElement).style.background = isStayOut ? "#1a1a1a" : "#0d0d0d";
      }}
    >
      <div style={{ fontWeight: "bold", letterSpacing: 1, fontSize: 13 }}>{opt.label}</div>
      <div style={{ color: "#666", fontSize: 11, marginTop: 3 }}>{opt.description}</div>
    </button>
  );
}

interface Props {
  prompt: DecisionPrompt;
  onDecide: (carIdx: number, action: string) => void;
}

export function DecisionModal({ prompt, onDecide }: Props) {
  return (
    <div style={{
      position: "fixed",
      inset: 0,
      background: "rgba(0, 0, 0, 0.82)",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      zIndex: 100,
    }}>
      <div style={{
        background: "#111",
        border: "1px solid #2a2a2a",
        padding: "28px 32px",
        width: 420,
        maxWidth: "90vw",
      }}>
        <div style={{ color: "#e8002d", fontSize: 10, letterSpacing: 3, marginBottom: 10 }}>
          STRATEGY CALL — LAP {prompt.lap}
        </div>
        <div style={{ color: "#e0e0e0", fontSize: 18, marginBottom: 4 }}>
          {prompt.car_name}
        </div>
        <div style={{ color: "#555", fontSize: 11, marginBottom: 24, letterSpacing: 1 }}>
          ON {prompt.current_compound} ({prompt.tyre_age} LAPS) &nbsp;·&nbsp; WINDOW CLOSES LAP {prompt.window_closes}
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {prompt.options.map((opt) => (
            <OptionButton
              key={opt.action}
              opt={opt}
              onClick={() => onDecide(prompt.car_idx, opt.action)}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
