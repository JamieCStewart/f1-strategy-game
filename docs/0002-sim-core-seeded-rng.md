# ADR-0002: Simulation core with seeded randomness

Race engine to include:
    Live game loop
    Monte Carlo rollouts of races
    Model validation
    Replay feature (with seed for randomness)

`sim/` is a pure Python library. All randomness flows through a single injected, seeded RNG. A race outcome is a deterministic function of `(model_version, seed, decision_log)`. Bug reports include a seed. AI rollouts reuse the exact production engine, so the opponents' beliefs match the game's reality.

CI lint rules should be set up to ensure sim doesn't use any random elements.
