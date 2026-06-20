"""Phase 0.2 hierarchical Bayesian lap-time model.

Install the optional deps before using:
    pip install -e ".[model]"   (from sim/)

Sub-modules:
    etl         — FastF1 ingestion → cleaned DataFrame
    features    — fuel estimate, stint index, compound encoding
    lap_time    — PyMC model definition, fit(), sample_posterior()
    sc_hazard   — per-circuit SC deployment probability estimate
    artifact    — PosteriorDraws dataclass, save/load (.npz)
    pace_model  — BayesianPaceModel (implements PaceModel protocol)
    validate    — calibration coverage, RMSE, backtest helpers
"""
