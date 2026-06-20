"""PyMC hierarchical lap-time model (Phase 0.2, tier 1).

Model structure (ADR-0003, incremental tier):
  circuit_base[C]  ~ Normal(mu_base, sigma_base)   partial pooling over circuits
  compound_offset[3] ~ Normal([0, 0.7, 1.2], 1.0)  soft / medium / hard
  deg_lin[3]       ~ HalfNormal(0.15)
  deg_quad[3]      ~ HalfNormal(0.003)
  fuel_effect      ~ Normal(0.032, 0.01)
  sigma_lap[3]     ~ HalfNormal(0.5)               compound-specific noise

  mu = circuit_base[circuit_idx]
     + compound_offset[compound]
     + deg_lin[compound] * tyre_age
     + deg_quad[compound] * tyre_age^2
     + fuel_effect * fuel_kg

  lap_time_s ~ Normal(mu, sigma_lap[compound])

Tier 2 (future): add car-season and driver partial-pooling layers.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .artifact import PosteriorDraws
from .features import to_model_arrays


def build_model(
    arrays: dict[str, np.ndarray],
    circuit_map: dict[str, int],
) -> Any:
    """Construct and return the PyMC model.

    Args:
        arrays: output of features.to_model_arrays().
        circuit_map: {circuit_id: int_index} from features.encode_circuits().

    Returns:
        A pm.Model instance (not yet sampled).
    """
    try:
        import pymc as pm  # type: ignore[import-untyped]
        import pytensor.tensor as pt  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "pymc is required. Install with: pip install -e '.[model]'"
        ) from exc

    n_circuits = len(circuit_map)
    circuit_idx = arrays["circuit_idx"]
    compound = arrays["compound"]
    tyre_age = arrays["tyre_age"]
    fuel_kg = arrays["fuel_kg"]
    lap_time_s = arrays["lap_time_s"]

    with pm.Model() as model:
        # --- circuit base time (partial pooling) ---
        mu_base = pm.Normal("mu_base", mu=90.0, sigma=10.0)
        sigma_base = pm.HalfNormal("sigma_base", sigma=5.0)
        circuit_base = pm.Normal(
            "circuit_base", mu=mu_base, sigma=sigma_base, shape=n_circuits
        )

        # --- compound-level parameters ---
        compound_offset = pm.Normal(
            "compound_offset", mu=[0.0, 0.7, 1.2], sigma=1.0, shape=3
        )
        deg_lin = pm.HalfNormal("deg_lin", sigma=0.15, shape=3)
        deg_quad = pm.HalfNormal("deg_quad", sigma=0.003, shape=3)

        # --- fuel effect ---
        fuel_effect = pm.Normal("fuel_effect", mu=0.032, sigma=0.01)

        # --- compound-specific observation noise ---
        sigma_lap = pm.HalfNormal("sigma_lap", sigma=0.5, shape=3)

        # --- likelihood ---
        mu = (
            circuit_base[circuit_idx]
            + compound_offset[compound]
            + deg_lin[compound] * tyre_age
            + deg_quad[compound] * tyre_age ** 2
            + fuel_effect * fuel_kg
        )
        pm.Normal("lap_time_obs", mu=mu, sigma=sigma_lap[compound], observed=lap_time_s)

    return model


def fit(
    arrays: dict[str, np.ndarray],
    circuit_map: dict[str, int],
    *,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
    random_seed: int = 0,
    progressbar: bool = True,
) -> Any:
    """Sample the posterior and return an ArviZ InferenceData object."""
    try:
        import pymc as pm  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "pymc is required. Install with: pip install -e '.[model]'"
        ) from exc

    model = build_model(arrays, circuit_map)
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=progressbar,
        )
    return idata


def extract_draws(
    idata: Any,
    circuit_id: str,
    circuit_map: dict[str, int],
    *,
    n_draws: int = 2000,
    rng: np.random.Generator | None = None,
) -> PosteriorDraws:
    """Pull N posterior draws for a given circuit into a PosteriorDraws artifact.

    Flattens chain × draw dimensions and samples without replacement where
    possible. This is the pre-race caching step.
    """
    if circuit_id not in circuit_map:
        raise KeyError(f"Circuit '{circuit_id}' not in circuit_map")
    cidx = circuit_map[circuit_id]
    if rng is None:
        rng = np.random.default_rng(0)

    post = idata.posterior

    def flat(var: str) -> np.ndarray:
        arr = post[var].values  # (chains, draws, ...)
        return arr.reshape(-1, *arr.shape[2:])

    all_n = flat("circuit_base").shape[0]
    replace = all_n < n_draws
    chosen = rng.choice(all_n, size=n_draws, replace=replace)

    return PosteriorDraws(
        circuit_id=circuit_id,
        circuit_base=flat("circuit_base")[chosen, cidx],
        compound_offset=flat("compound_offset")[chosen],
        deg_lin=flat("deg_lin")[chosen],
        deg_quad=flat("deg_quad")[chosen],
        fuel_effect=flat("fuel_effect")[chosen],
        noise_sd=flat("sigma_lap")[chosen],
    )
