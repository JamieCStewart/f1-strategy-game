"""FastF1 ingestion: load a race session and return a clean lap DataFrame.

The public surface is two functions:
  - clean_laps(raw_laps, circuit_id, year) — pure filter/transform, testable
    without a network connection.
  - load_race_laps(year, race, cache_dir) — calls FastF1 then delegates to
    clean_laps.

ADR-0002: no I/O in sim/; ETL is explicitly outside that boundary.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    pass

# FastF1 string → our Compound int codes (matches config.Compound enum)
_COMPOUND_MAP: dict[str, int] = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}

# TrackStatus values that indicate a clean, green-flag lap
_GREEN_STATUS = {"1"}


def clean_laps(
    raw_laps: pd.DataFrame,
    circuit_id: str,
    year: int,
) -> pd.DataFrame:
    """Filter and featurise a raw FastF1 laps DataFrame.

    Keeps only clean, green-flag, dry-compound laps. Returns a DataFrame with
    columns:
        circuit_id, year, driver_id, team_id, lap_number,
        compound (int), tyre_age (float), stint_index (int),
        fuel_kg_est (float), lap_time_s (float)
    """
    df = raw_laps.copy()

    # --- filter ---
    # valid lap time
    df = df[df["LapTime"].notna()]
    # green flag only
    df = df[df["TrackStatus"].isin(_GREEN_STATUS)]
    # not an in-lap or out-lap
    df = df[df["PitInTime"].isna() & df["PitOutTime"].isna()]
    # dry compounds only
    df = df[df["Compound"].isin(_COMPOUND_MAP)]

    if df.empty:
        return _empty_output()

    # --- transform ---
    out = pd.DataFrame()
    out["circuit_id"] = circuit_id
    out["year"] = year
    out["driver_id"] = df["Driver"].values
    out["team_id"] = df["Team"].values
    out["lap_number"] = df["LapNumber"].astype(int).values
    out["compound"] = df["Compound"].map(_COMPOUND_MAP).astype(int).values
    out["tyre_age"] = df["TyreLife"].astype(float).values
    out["lap_time_s"] = df["LapTime"].dt.total_seconds().values

    # stint index per driver (number of stops taken before this lap)
    stint_idx = (
        df.groupby("Driver")["Stint"].transform(lambda s: s.rank(method="dense") - 1)
        if "Stint" in df.columns
        else _infer_stint_index(df)
    )
    out["stint_index"] = stint_idx.astype(int).values

    # fuel estimate: 105 kg at lap 1, burning 1.8 kg/lap
    out["fuel_kg_est"] = (105.0 - (out["lap_number"] - 1) * 1.8).clip(lower=0.0)

    return out.reset_index(drop=True)


def _infer_stint_index(df: pd.DataFrame) -> pd.Series:
    """Fallback stint index when FastF1 'Stint' column is absent."""
    result = pd.Series(0, index=df.index)
    for driver, grp in df.groupby("Driver"):
        prev_compound = grp["Compound"].values[0]
        stint = 0
        indices = []
        for i, (_, row) in enumerate(grp.iterrows()):
            if i > 0 and row["Compound"] != prev_compound:
                stint += 1
            prev_compound = row["Compound"]
            indices.append((row.name, stint))
        for idx, s in indices:
            result.at[idx] = s
    return result


def _empty_output() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "circuit_id", "year", "driver_id", "team_id", "lap_number",
            "compound", "tyre_age", "stint_index", "fuel_kg_est", "lap_time_s",
        ]
    )


def load_race_laps(
    year: int,
    race: str,
    *,
    cache_dir: str | None = None,
    circuit_id: str | None = None,
) -> pd.DataFrame:
    """Load and clean laps for one race weekend from FastF1.

    Args:
        year: season year, e.g. 2024.
        race: race name or round number accepted by FastF1, e.g. "Barcelona".
        cache_dir: path for FastF1's file cache. Defaults to the
            F1_CACHE_DIR env-var, then ~/.cache/fastf1.
        circuit_id: override the circuit identifier stored in the output.
            Defaults to the FastF1 event location string.
    """
    try:
        import fastf1  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "fastf1 is required for ETL. Install with: pip install -e '.[model]'"
        ) from exc

    resolved_cache = cache_dir or os.environ.get(
        "F1_CACHE_DIR", os.path.expanduser("~/.cache/fastf1")
    )
    os.makedirs(resolved_cache, exist_ok=True)
    fastf1.Cache.enable_cache(resolved_cache)

    session = fastf1.get_session(year, race, "R")
    session.load(laps=True, weather=False, telemetry=False, messages=False)

    resolved_circuit = circuit_id or str(session.event["Location"])
    return clean_laps(session.laps, circuit_id=resolved_circuit, year=year)
