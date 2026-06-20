"""Per-circuit safety-car deployment probability estimator.

Fits a Beta-Binomial model to the historical SC rate for a circuit and
returns the posterior mean probability per lap. The result is used to
update CircuitConfig.sc_prob_per_lap before a race, replacing the global
default.

Model:
    p ~ Beta(alpha_prior, beta_prior)       non-informative prior
    sc_laps | p, n ~ Binomial(n, p)         observed data
    E[p | data] = (sc_laps + alpha) / (n + alpha + beta)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# Weakly informative prior: ~1.2 % per lap (based on historical average)
_ALPHA_PRIOR = 1.2
_BETA_PRIOR = 98.8


def estimate_sc_prob(
    df_raw: pd.DataFrame,
    circuit_id: str,
    *,
    alpha_prior: float = _ALPHA_PRIOR,
    beta_prior: float = _BETA_PRIOR,
) -> float:
    """Return the posterior mean SC probability per lap for a circuit.

    Args:
        df_raw: raw FastF1 laps DataFrame (before clean_laps filtering) so
                SC-flag laps are still present.
        circuit_id: used only for error messages; caller has already filtered
                    to the desired circuit.
        alpha_prior: Beta prior alpha parameter.
        beta_prior: Beta prior beta parameter.
    """
    if "TrackStatus" not in df_raw.columns:
        raise ValueError("df_raw must contain a 'TrackStatus' column")

    total_laps = len(df_raw)
    if total_laps == 0:
        raise ValueError(f"No laps found for circuit '{circuit_id}'")

    # TrackStatus "4" = safety car, "5" = virtual SC, "6" = VSC ending
    sc_mask = df_raw["TrackStatus"].isin({"4", "5", "6"})
    sc_laps = int(sc_mask.sum())

    # Beta-Binomial posterior mean
    return (sc_laps + alpha_prior) / (total_laps + alpha_prior + beta_prior)


def estimate_sc_prob_from_sessions(
    sessions: list[pd.DataFrame],
    circuit_id: str,
) -> float:
    """Pool SC observations across multiple seasons for more stable estimates."""
    if not sessions:
        raise ValueError("sessions list is empty")
    combined = pd.concat(sessions, ignore_index=True)
    return estimate_sc_prob(combined, circuit_id)
