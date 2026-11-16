"""Consistent wall-clock timing campaign for the revised SC-CAIS manuscript.

Boundary: start immediately before observation reconstruction, stop immediately
after the assurance decision and before offline true-state audit. Policy inference
and physical plant execution are excluded. Per-action times include observation
reconstruction, checks, attempted repair, verification, and fallback selection.

Run (after installing project):
  python examples/revision_latency_campaign.py --output outputs/revision_latency
  python examples/revision_latency_campaign.py --quick --output outputs/revision_latency_smoke

All eight observed-only control modes are measured with matched scenarios and
seeds. Results are runtime- and hardware-specific, not real-time guarantees.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sc_cais.revision_experiments import run_campaign, revision_scenarios
from sc_cais.scenarios import default_scenarios


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-mode assurance timing")
    parser.add_argument("--output", type=Path, default=Path("outputs/revision_latency"))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    all_modes = (
        ("observed", "rule_repair"),
        ("observed", "monitor_only"),
        ("observed", "reject_only"),
        ("observed", "quantity_clip"),
        ("observed", "no_trust"),
        ("observed", "no_authority"),
        ("observed", "no_route_repair"),
        ("observed", "no_quantity_clipping"),
    )
    scenarios = (
        default_scenarios()[:1]
        if args.quick
        else [*default_scenarios(), revision_scenarios()[-1]]
    )
    seeds = (0,) if args.quick else tuple(range(10))
    horizon = 3 if args.quick else 36
    runs, actions = run_campaign(
        seeds=seeds, horizon=horizon, scenarios=scenarios,
        modes_override=all_modes,
        output_dir=args.output,
    )
    if not args.quick:
        assert len(runs) == 11 * 10 * 3 * len(all_modes)

    raw = actions["assurance_latency_ms"].to_numpy(dtype=float)
    assert len(raw) > 0 and np.all(np.isfinite(raw)) and np.all(raw > 0), (
        "All modes must record positive decision latency"
    )

    # Summarize at the action level and retain raw measurements for re-analysis.
    rows = []
    for (policy, mode), sub in actions.groupby(["policy", "repair_mode"], sort=True):
        values = sub["assurance_latency_ms"].to_numpy(dtype=float)
        rows.append({
            "policy": policy, "repair_mode": mode, "n_actions": len(values),
            "mean_latency_ms": float(np.mean(values)),
            "median_latency_ms": float(np.median(values)),
            "p95_latency_ms": float(np.percentile(values, 95)),
            "p99_latency_ms": float(np.percentile(values, 99)),
        })
    table = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    table_file = args.output / "latency_by_policy_mode.csv"
    table.to_csv(table_file, index=False)
    summary_file = args.output / "revision_summary.csv"
    actions_file = args.output / "revision_actions.csv"
    manifest = {
        "protocol": "observation reconstruction + assurance decision, excluding policy inference, simulator step and offline audit",
        "clock": "time.perf_counter_ns",
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "quick": args.quick,
        "seeds": list(seeds),
        "horizon": horizon,
        "scenarios": [s.name for s in scenarios],
        "runs": len(runs),
        "actions": len(actions),
        "sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (table_file, summary_file, actions_file)
        },
    }
    (args.output / "latency_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(table.to_string(index=False))
    print(f"Completed {len(runs)} runs, {len(actions)} timed decisions")


if __name__ == "__main__":
    main()
