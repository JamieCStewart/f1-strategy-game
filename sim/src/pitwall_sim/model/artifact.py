"""PosteriorDraws: the pre-race model artifact shipped to the sim engine.

Saves/loads a flat collection of posterior draws for one circuit. The engine
indexes into these arrays at runtime — no PyMC, no ArviZ required to run a
race.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PosteriorDraws:
    """N pre-sampled draws from the posterior for a specific circuit.

    All arrays have leading dimension N (number of draws).
    """

    circuit_id: str
    circuit_base: np.ndarray     # (N,)       base lap time for this circuit
    compound_offset: np.ndarray  # (N, 3)     soft / medium / hard pace offset
    deg_lin: np.ndarray          # (N, 3)     linear degradation s/lap
    deg_quad: np.ndarray         # (N, 3)     quadratic degradation s/lap^2
    fuel_effect: np.ndarray      # (N,)       s per kg of fuel
    noise_sd: np.ndarray         # (N, 3)     per-compound observation noise

    @property
    def n_draws(self) -> int:
        return len(self.circuit_base)

    def save(self, path: str | Path) -> None:
        """Persist to a .npz file."""
        np.savez_compressed(
            str(path),
            circuit_id=np.array([self.circuit_id]),
            circuit_base=self.circuit_base,
            compound_offset=self.compound_offset,
            deg_lin=self.deg_lin,
            deg_quad=self.deg_quad,
            fuel_effect=self.fuel_effect,
            noise_sd=self.noise_sd,
        )

    @classmethod
    def load(cls, path: str | Path) -> "PosteriorDraws":
        """Load from a .npz file."""
        data = np.load(str(path))
        return cls(
            circuit_id=str(data["circuit_id"][0]),
            circuit_base=data["circuit_base"],
            compound_offset=data["compound_offset"],
            deg_lin=data["deg_lin"],
            deg_quad=data["deg_quad"],
            fuel_effect=data["fuel_effect"],
            noise_sd=data["noise_sd"],
        )
