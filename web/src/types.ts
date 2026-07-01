export interface CarState {
  car_idx: number;
  car_name: string;
  position: number;
  gap_to_leader: number;
  interval: number;
  compound: "S" | "M" | "H";
  tyre_age: number;
  last_lap_time: number;
  cumulative_time: number;
  pitted_this_lap: boolean;
  is_player: boolean;
}

export interface DecisionOption {
  action: string;
  label: string;
  description: string;
}

export interface DecisionPrompt {
  type: "decision_prompt";
  lap: number;
  car_idx: number;
  car_name: string;
  tyre_age: number;
  current_compound: string;
  options: DecisionOption[];
  window_closes: number;
}

export interface ClassificationEntry {
  position: number;
  car_idx: number;
  car_name: string;
  total_time: number;
  gap: number;
  is_player: boolean;
}

export interface RacingState {
  phase: "racing";
  lap: number;
  totalLaps: number;
  cars: CarState[];
  scActive: boolean;
  playerCarIndices: number[];
}

export interface DecisionState {
  phase: "decision";
  lap: number;
  totalLaps: number;
  cars: CarState[];
  scActive: boolean;
  playerCarIndices: number[];
  prompt: DecisionPrompt;
}

export interface FinishedState {
  phase: "finished";
  classification: ClassificationEntry[];
  playerCarIndices: number[];
}

export type AppState =
  | { phase: "lobby" }
  | RacingState
  | DecisionState
  | FinishedState;
