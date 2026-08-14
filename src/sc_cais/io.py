"""Serialization helpers for reproducibility outputs."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from sc_cais.models import StepResult


def write_history_json(results: list[StepResult], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: list[dict[str, Any]] = []
    for result in results:
        item = asdict(result)
        for action in item["proposed_plan"]["actions"]:
            action["action_type"] = str(action["action_type"])
        for i, outcome in enumerate(result.outcomes):
            encoded = item["outcomes"][i]
            encoded["decision"] = outcome.decision.value
            encoded["proposed_action"]["action_type"] = outcome.proposed_action.action_type.value
            encoded["executed_action"]["action_type"] = outcome.executed_action.action_type.value
            for j, violation in enumerate(outcome.violations):
                encoded["violations"][j]["severity"] = violation.severity.value
            for j, violation in enumerate(outcome.residual_violations):
                encoded["residual_violations"][j]["severity"] = violation.severity.value
        payload.append(item)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target
