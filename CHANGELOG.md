# Changelog

## 0.3.2

- Maintenance-only typing fix for Matplotlib Figure/SubFigure inference in article output generation.
- No changes to simulation logic, scenarios, policies, seeds, metrics, or experimental design.

## 0.3.1

- Quality-only maintenance release; experimental semantics and frozen campaign parameters are unchanged.
- Resolved Ruff findings across source, examples, and tests.
- Resolved strict mypy typing issues in internal protocols, benchmark interfaces, plotting helpers, and environment APIs.
- Added mypy overrides for third-party libraries that do not ship complete inline typing in the supported environment.
- Confirmed 20/20 tests, verification smoke test, and quick publication pipeline.
- Confirmed quick-run scientific outputs are identical to v0.3.0 except wall-clock latency fields.

## 0.3.0
- Added seed-paired Wilcoxon inference, bootstrap confidence intervals and Cohen's dz.
- Added step-level trajectory logging with pre/disruption/recovery phase labels.
- Added repair-vs-fallback ablation outputs.
- Added disruption-degradation-recovery article figures.
- Added frozen final experiment runner for 10 seeds and 36 steps.
- Expanded verification to the v0.3 experimental pipeline.


## 0.1.0 - 2026-08-12

- Initial research-grade SC-CAIS library.
- Multi-echelon supply-chain simulator with action plans and shipment lead times.
- Policy-agnostic runtime assurance with approve, repair and fallback semantics.
- Physical, governance and cyber-data trust constraints.
- Trust-aware source/route repair preserving agent intent when possible.
- Operational, cyber and compound disruption scenarios.
- Rule-based, cost-aware optimizer, adaptive agentic and stochastic policies.
- Reproducible multi-seed benchmark runner, severity grids and paired statistics.
- Article-ready CSV aggregation, frontier figure and SHA-256 environment manifest.
- CLI, examples, documentation, tests and CI configuration.

## 0.2.0

- Added rolling-horizon MILP baseline using SciPy `milp`.
- Added deterministic observation uncertainty and a severity × uncertainty campaign design.
- Added instantaneous service-level metrics, resilience loss area, maximum degradation, and time-to-recovery.
- Added article-ready resilience and robustness figures/tables.
- Added publication-scale and fast reference campaign scripts.
- Extended reproducibility metadata with SciPy versioning.
