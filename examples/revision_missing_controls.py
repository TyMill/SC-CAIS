"""Run only the editor's two missing paired controls.

Usage:
  python examples/revision_missing_controls.py \
      --output outputs/revision_missing

This does NOT rerun the original 2640-run campaign. It adds 330 native
monitor-only runs and 330 reverse-order SC-CAIS runs with the same 11 scenarios,
10 seeds, 36 time steps and FIFO on-time service accounting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from sc_cais.revision_experiments import run_campaign, revision_scenarios
from sc_cais.scenarios import default_scenarios


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/revision_missing"))
    parser.add_argument("--quick", action="store_true", help="1 seed, 4 steps, normal scenario")
    args = parser.parse_args()
    scenarios = (
        default_scenarios()[:1]
        if args.quick
        else [*default_scenarios(), revision_scenarios()[-1]]
    )
    seeds = (0,) if args.quick else tuple(range(10))
    horizon = 4 if args.quick else 36

    controls = (
        ("monitor_native", (("observed", "monitor_only"),), "native"),
        ("repair_reverse", (("observed", "rule_repair"),), "reverse"),
    )
    sum_parts: list[pd.DataFrame] = []
    action_parts: list[pd.DataFrame] = []
    for label, modes, order in controls:
        summary, actions = run_campaign(
            scenarios=scenarios,
            seeds=seeds,
            horizon=horizon,
            modes_override=modes,
            action_order=order,
        )
        summary.insert(0, "campaign", label)
        actions.insert(0, "campaign", label)
        sum_parts.append(summary)
        action_parts.append(actions)

    args.output.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / "missing_controls_summary.csv"
    actions_path = args.output / "missing_controls_actions.csv"
    pd.concat(sum_parts, ignore_index=True).to_csv(summary_path, index=False)
    pd.concat(action_parts, ignore_index=True).to_csv(actions_path, index=False)
    summary_rows = sum(len(part) for part in sum_parts)
    if not args.quick:
        assert summary_rows == 660, summary_rows
    manifest = {
        "horizon": horizon,
        "seeds": seeds,
        "scenarios": [scenario.name for scenario in scenarios],
        "run_count": summary_rows,
        "files": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (summary_path, actions_path)
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Completed {summary_rows} runs. Manifest: {args.output / 'manifest.json'}")


if __name__ == "__main__":
    main()
