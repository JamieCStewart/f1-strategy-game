"""Feature-engineering helpers that operate on the clean lap DataFrame.

All functions are pure transforms on DataFrames — no I/O, no FastF1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def encode_circuits(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Add a `circuit_idx` integer column; return the mapping dict."""
    circuits = sorted(df["circuit_id"].unique())
    mapping = {c: i for i, c in enumerate(circuits)}
    out = df.copy()
    out["circuit_idx"] = out["circuit_id"].map(mapping).astype(int)
    return out, mapping


def filter_outlier_laps(
    df: pd.DataFrame,
    *,
    low_quantile: float = 0.005,
    high_quantile: float = 0.995,
) -> pd.DataFrame:
    """Drop laps whose time is outside the [low, high] quantile band.

    Applied per-circuit so circuits with very different base times don't
    contaminate each other's thresholds.
    """
    mask = pd.Series(True, index=df.index)
    for _, grp in df.groupby("circuit_id"):
        lo = grp["lap_time_s"].quantile(low_quantile)
        hi = grp["lap_time_s"].quantile(high_quantile)
        mask.loc[grp.index] = grp["lap_time_s"].between(lo, hi)
    return df[mask].copy()


def train_test_split(
    df: pd.DataFrame,
    *,
    held_out_circuits: list[str] | None = None,
    held_out_years: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train / test by circuit or year.

    If neither argument is given, holds out the most recent year in each
    circuit as a default.
    """
    if held_out_circuits is not None:
        mask = df["circuit_id"].isin(held_out_circuits)
    elif held_out_years is not None:
        mask = df["year"].isin(held_out_years)
    else:
        # default: most recent year per circuit
        latest = df.groupby("circuit_id")["year"].transform("max")
        mask = df["year"] == latest
    return df[~mask].copy(), df[mask].copy()


def to_model_arrays(
    df: pd.DataFrame,
    circuit_map: dict[str, int],
) -> dict[str, np.ndarray]:
    """Convert a clean DataFrame to numpy arrays ready for PyMC.

    Requires `circuit_idx` column (add via encode_circuits first).
    """
    df = df.copy()
    if "circuit_idx" not in df.columns:
        df["circuit_idx"] = df["circuit_id"].map(circuit_map).astype(int)
    return {
        "circuit_idx": df["circuit_idx"].to_numpy(dtype=np.int64),
        "compound": df["compound"].to_numpy(dtype=np.int64),
        "tyre_age": df["tyre_age"].to_numpy(dtype=np.float64),
        "fuel_kg": df["fuel_kg_est"].to_numpy(dtype=np.float64),
        "lap_time_s": df["lap_time_s"].to_numpy(dtype=np.float64),
    }
