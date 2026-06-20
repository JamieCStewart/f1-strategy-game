# ADR-0009: PyMC as the Bayesian modelling framework

**Status:** Accepted (Phase 0.2)

ADR-0003 specified a hierarchical Bayesian lap-time model and deferred the
framework choice (PyMC vs NumPyro) to a benchmark spike. Having weighed the
trade-offs, PyMC is chosen for Phase 0.2.

## Decision

Use **PyMC ≥ 5** for model definition, NUTS sampling, and ArviZ-based
diagnostics. Training runs offline; the runtime engine (`sim/`) never imports
PyMC — it only loads pre-drawn posterior arrays from `.npz` artifacts.

## Rationale

- **Readable model syntax.** PyMC's context-manager style maps directly to the
  notation in the architecture doc and ADRs, keeping the code self-documenting
  and easy for non-ML contributors to audit.
- **ArviZ integration.** Built-in convergence diagnostics (R-hat, ESS),
  calibration plots, and posterior predictive checks are critical for the CI
  validation gate (§5.3 of the architecture doc).
- **Runtime cost is irrelevant.** The engine indexes into pre-drawn numpy
  arrays, so PyMC's slower sampling speed vs. NumPyro/JAX does not affect
  race-time performance at all.
- **Maturity for hierarchical models.** Partial pooling, non-centred
  parameterisations, and heteroskedastic likelihoods are well-trodden territory
  in the PyMC ecosystem with extensive documentation and worked examples.

## Trade-offs / when to revisit

- If training time on multiple seasons of data becomes a bottleneck (hours, not
  minutes), migrate the model definition to NumPyro. The `PaceModel` protocol
  contract and `.npz` artifact format are framework-agnostic; swapping
  frameworks is a training concern only.
- NumPyro also becomes the natural choice if the model grows to include
  GPU-accelerated inference (e.g., live posterior updating in ADR-0011).
