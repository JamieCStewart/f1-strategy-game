"""Tests for etl.clean_laps — no FastF1 calls."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pitwall_sim.model.etl import clean_laps


def _raw_laps(n: int = 20, seed: int = 0) -> pd.DataFrame:
    """Minimal synthetic FastF1-shaped DataFrame."""
    rng = np.random.default_rng(seed)
    base_time = pd.Timedelta(seconds=91.0)
    noise = [pd.Timedelta(seconds=float(rng.normal(0, 0.3))) for _ in range(n)]

    return pd.DataFrame(
        {
            "LapTime": [base_time + d for d in noise],
            "Compound": rng.choice(["SOFT", "MEDIUM", "HARD"], size=n),
            "TyreLife": rng.integers(1, 20, size=n),
            "TrackStatus": ["1"] * n,
            "PitInTime": pd.to_timedelta([pd.NaT] * n),
            "PitOutTime": pd.to_timedelta([pd.NaT] * n),
            "Driver": [f"D{i % 5:02d}" for i in range(n)],
            "Team": [f"T{i % 3}" for i in range(n)],
            "LapNumber": list(range(1, n + 1)),
            "Stint": [1] * (n // 2) + [2] * (n - n // 2),
        }
    )


def test_clean_laps_output_schema():
    df = clean_laps(_raw_laps(), circuit_id="Barcelona", year=2024)
    expected = {
        "circuit_id", "year", "driver_id", "team_id", "lap_number",
        "compound", "tyre_age", "stint_index", "fuel_kg_est", "lap_time_s",
    }
    assert expected.issubset(set(df.columns))


def test_clean_laps_removes_sc_laps():
    raw = _raw_laps(10)
    raw.loc[3:5, "TrackStatus"] = "4"
    df = clean_laps(raw, circuit_id="Barcelona", year=2024)
    assert len(df) == 7


def test_clean_laps_removes_in_out_laps():
    raw = _raw_laps(10)
    raw.loc[2, "PitInTime"] = pd.Timedelta(seconds=3600)
    raw.loc[4, "PitOutTime"] = pd.Timedelta(seconds=3600)
    df = clean_laps(raw, circuit_id="Barcelona", year=2024)
    assert len(df) == 8


def test_clean_laps_removes_wet_compounds():
    raw = _raw_laps(10)
    raw.loc[0:1, "Compound"] = "INTERMEDIATE"
    df = clean_laps(raw, circuit_id="Barcelona", year=2024)
    assert len(df) == 8
    assert (df["compound"].between(0, 2)).all()


def test_clean_laps_fuel_estimate_decreases():
    raw = _raw_laps(20)
    df = clean_laps(raw, circuit_id="Barcelona", year=2024)
    fuel = df.sort_values("lap_number")["fuel_kg_est"].values
    assert (fuel[:-1] >= fuel[1:]).all()


def test_clean_laps_removes_nan_lap_times():
    raw = _raw_laps(10)
    raw.loc[5, "LapTime"] = pd.NaT
    df = clean_laps(raw, circuit_id="Barcelona", year=2024)
    assert len(df) == 9


def test_clean_laps_empty_input_returns_empty():
    raw = _raw_laps(5)
    raw["TrackStatus"] = "4"  # all SC
    df = clean_laps(raw, circuit_id="Barcelona", year=2024)
    assert len(df) == 0
    assert "lap_time_s" in df.columns
