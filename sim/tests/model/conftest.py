"""Synthetic fixtures for model tests — no FastF1, no network."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pitwall_sim.model.artifact import PosteriorDraws


def synthetic_laps(
    *,
    n_laps: int = 300,
    circuits: list[str] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a clean-lap DataFrame matching the schema from etl.clean_laps."""
    rng = np.random.default_rng(seed)
    if circuits is None:
        circuits = ["Barcelona", "Monza"]

    rows = []
    for i in range(n_laps):
        circuit = circuits[i % len(circuits)]
        base = 82.0 if circuit == "Monza" else 91.0
        compound = int(rng.integers(0, 3))
        tyre_age = float(rng.integers(1, 25))
        comp_offsets = [0.0, 0.7, 1.2]
        deg = [0.08, 0.05, 0.03]
        lap_number = int(rng.integers(3, 55))
        fuel_kg = max(0.0, 105.0 - (lap_number - 1) * 1.8)
        mu = (
            base
            + comp_offsets[compound]
            + deg[compound] * tyre_age
            + 0.032 * fuel_kg
            + rng.normal(0.0, 0.3)
        )
        rows.append(
            {
                "circuit_id": circuit,
                "year": 2024,
                "driver_id": f"D{i % 10:02d}",
                "team_id": f"T{i % 5}",
                "lap_number": lap_number,
                "compound": compound,
                "tyre_age": tyre_age,
                "stint_index": int(rng.integers(0, 3)),
                "fuel_kg_est": fuel_kg,
                "lap_time_s": float(max(mu, base * 0.9)),
            }
        )
    return pd.DataFrame(rows)


def synthetic_draws(
    *,
    circuit_id: str = "Barcelona",
    n_draws: int = 50,
    seed: int = 0,
) -> PosteriorDraws:
    """Minimal PosteriorDraws for tests that don't need realistic values."""
    rng = np.random.default_rng(seed)
    return PosteriorDraws(
        circuit_id=circuit_id,
        circuit_base=rng.normal(91.0, 0.5, size=n_draws),
        compound_offset=rng.normal([[0.0, 0.7, 1.2]], 0.1, size=(n_draws, 3)),
        deg_lin=np.abs(rng.normal(0.06, 0.01, size=(n_draws, 3))),
        deg_quad=np.abs(rng.normal(0.001, 0.0002, size=(n_draws, 3))),
        fuel_effect=rng.normal(0.032, 0.002, size=n_draws),
        noise_sd=np.abs(rng.normal(0.3, 0.05, size=(n_draws, 3))),
    )


@pytest.fixture
def laps_df() -> pd.DataFrame:
    return synthetic_laps()


@pytest.fixture
def draws() -> PosteriorDraws:
    return synthetic_draws()
