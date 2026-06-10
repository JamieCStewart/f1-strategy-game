# ADR-0006: Local-first development with ephemeral GKE environments

## Context

A goal of the project is showcase-grade infrastructure (Kubernetes, IaC, observability, load testing) on a personal budget. GCP's new-customer credits ($300 / 90 days as of mid-2026) comfortably exceed AWS's current offer, and GKE Autopilot is the lowest-friction managed Kubernetes for a solo developer. However, an always-on cluster with managed Postgres and Redis would burn those credits in weeks.

## Decision

- All day-to-day development runs locally: `docker-compose` for the full stack, `kind` for validating Kubernetes manifests.
- All cloud infrastructure is defined in Terraform (GKE Autopilot, Memorystore, Cloud SQL, Artifact Registry, LB) and deployed via Helm — and is **ephemeral**: created for demo windows and load tests, destroyed afterward. Environment lifetime is a documented runbook (`make env-up` / `make env-down`).
- A minimal always-on demo runs on Cloud Run (scale-to-zero) so the project has a permanent live link at near-zero cost.
- Budget alerts are configured in Terraform before any other resource.

## Consequences

- The infrastructure *code* — the actual showcase artifact — exists and is exercised regularly, while cloud spend stays within free credits.
- Load-test results ("N concurrent race rooms per node") are produced during planned windows and committed as reports, with the environment that produced them reproducible from a commit hash.
- Trade-off: no permanently-live Kubernetes deployment; mitigated by the Cloud Run demo and by recorded demos. Revisit if the game gains a real user base (that ADR would be a nice problem to have).
