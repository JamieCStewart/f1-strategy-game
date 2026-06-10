# ADR-0003: Hierarchical Bayesian lap-time model

The lap-time model must: 
- drive a fully stochastic simulation
- generalize across circuits and seasons including a new circuit or a driver's first season with a team 

Model lap time with a hierarchical Bayesian regression (partial pooling over circuit, car–season, driver, and compound effects; nonlinear degradation curves; fuel and track-evolution terms; heteroskedastic residual noise). Sampling from the posterior *is* the simulation's pace draw. Framework choice (PyMC vs. NumPyro) is deferred to a benchmark spike and its own ADR. Companion models: per-circuit safety-car hazard; weather process (phase 1.2).

- Uncertainty bands shown to the player are the model's actual posterior, not a cosmetic ±.
- Partial pooling gives sane behavior in low-data regimes instead of wild extrapolation.
- Interpretable effects double as documentation ("our model thinks the soft gives up X s/lap after N laps at Circuit Y") and as validation targets.
- Costs: slower training than GBM; inference engineering needed so posterior sampling is fast enough for MC rollouts (we will cache posterior draws per race).
- Calibration backtests become a CI promotion gate for model artifacts.
