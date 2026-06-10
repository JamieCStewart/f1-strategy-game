# ADR-0005: Roster as a configuration layer; fictional names by default

## Context

The model trains on real historical timing data, so the most natural presentation uses real team and driver names. But those names, and F1's brand assets, are trademarked, and the rights holders are known to be protective even of non-commercial fan projects once publicly hosted. This is a personal, non-commercial project; we want to minimize takedown risk without weakening the data-science work. (Risk reduction, not legal advice.)

## Decision

The simulation and model operate on opaque car/driver IDs only. Display names, liveries, and team identities come from a roster configuration resolved at the presentation layer (`ROSTER=fictional|real`). Public deployments ship `fictional`; `real` exists for private demos. No F1 logos, wordmarks, or official typefaces are used anywhere, and the product name avoids "F1"/"Formula 1". Circuit layouts and the historical data pipeline are unaffected.

## Consequences

- A legal repaint is a config change, not a refactor.
- The model documentation can still honestly describe training on real historical data without the *game* trading on protected identities.
- Minor cost: a small content task to invent a fictional grid (also a fun one).
