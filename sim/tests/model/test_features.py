"""Tests for feature engineering helpers."""
from __future__ import annotations

import numpy as np
import pytest

from pitwall_sim.model.features import (
    encode_circuits,
    filter_outlier_laps,
    to_model_arrays,
    train_test_split,
)

from .conftest import synthetic_laps


def test_encode_circuits_assigns_contiguous_ints():
    df = synthetic_laps(circuits=["Barcelona", "Monza", "Spa"])
    df_enc, mapping = encode_circuits(df)
    assert set(mapping.keys()) == {"Barcelona", "Monza", "Spa"}
    assert sorted(mapping.values()) == [0, 1, 2]
    assert df_enc["circuit_idx"].between(0, 2).all()


def test_encode_circuits_deterministic():
    df = synthetic_laps()
    _, m1 = encode_circuits(df)
    _, m2 = encode_circuits(df)
    assert m1 == m2


def test_filter_outlier_laps_removes_extremes():
    df = synthetic_laps(n_laps=200)
    df.loc[0, "lap_time_s"] = 1.0    # absurdly fast
    df.loc[1, "lap_time_s"] = 999.0  # absurdly slow
    filtered = filter_outlier_laps(df)
    assert len(filtered) < len(df)
    assert filtered["lap_time_s"].max() < 999.0
    assert filtered["lap_time_s"].min() > 1.0


def test_train_test_split_by_year():
    df = synthetic_laps(n_laps=100)
    df.loc[:49, "year"] = 2023
    df.loc[50:, "year"] = 2024
    train, test = train_test_split(df, held_out_years=[2024])
    assert set(train["year"].unique()) == {2023}
    assert set(test["year"].unique()) == {2024}
    assert len(train) + len(test) == len(df)


def test_train_test_split_default_holds_out_latest_per_circuit():
    df = synthetic_laps(n_laps=100, circuits=["Barcelona", "Monza"])
    df.loc[:49, "year"] = 2022
    df.loc[50:, "year"] = 2024
    train, test = train_test_split(df)
    assert test["year"].unique().tolist() == [2024]


def test_to_model_arrays_shapes():
    df = synthetic_laps(n_laps=50)
    df_enc, mapping = encode_circuits(df)
    arrays = to_model_arrays(df_enc, mapping)
    n = len(df)
    for key in ("circuit_idx", "compound", "tyre_age", "fuel_kg", "lap_time_s"):
        assert arrays[key].shape == (n,), f"{key} wrong shape"


def test_to_model_arrays_compound_range():
    df = synthetic_laps(n_laps=100)
    df_enc, mapping = encode_circuits(df)
    arrays = to_model_arrays(df_enc, mapping)
    assert arrays["compound"].min() >= 0
    assert arrays["compound"].max() <= 2
