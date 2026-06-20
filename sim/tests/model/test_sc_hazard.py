"""Tests for the SC hazard model."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pitwall_sim.model.sc_hazard import estimate_sc_prob, estimate_sc_prob_from_sessions


def _laps(n: int, n_sc: int) -> pd.DataFrame:
    statuses = ["4"] * n_sc + ["1"] * (n - n_sc)
    return pd.DataFrame({"TrackStatus": statuses})


def test_zero_sc_laps_returns_prior_mean():
    df = _laps(n=100, n_sc=0)
    p = estimate_sc_prob(df, "Barcelona")
    # (0 + 1.2) / (100 + 1.2 + 98.8) = 1.2 / 200 = 0.006
    assert p == pytest.approx(0.006, abs=0.001)


def test_high_sc_rate_gives_higher_estimate():
    low_sc = estimate_sc_prob(_laps(100, 2), "Barcelona")
    high_sc = estimate_sc_prob(_laps(100, 20), "Barcelona")
    assert high_sc > low_sc


def test_pooling_more_sessions_stabilises_estimate():
    # More laps → prior has less influence → estimate moves toward observed rate
    s1 = _laps(100, 5)
    s2 = _laps(100, 5)
    p1 = estimate_sc_prob(s1, "Monza")
    p_pooled = estimate_sc_prob_from_sessions([s1, s2], "Monza")
    # Both sessions have identical 5% SC rate; pooled estimate converges toward
    # 5% (away from the prior mean ~0.6%), so pooled > single-session estimate.
    observed_rate = 5 / 100
    assert abs(p_pooled - observed_rate) < abs(p1 - observed_rate)


def test_missing_track_status_raises():
    df = pd.DataFrame({"LapTime": [90.0, 91.0]})
    with pytest.raises(ValueError, match="TrackStatus"):
        estimate_sc_prob(df, "Barcelona")


def test_empty_dataframe_raises():
    df = pd.DataFrame({"TrackStatus": []})
    with pytest.raises(ValueError, match="No laps"):
        estimate_sc_prob(df, "Barcelona")
