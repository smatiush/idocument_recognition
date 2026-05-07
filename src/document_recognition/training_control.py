from __future__ import annotations

import json
from pathlib import Path
import time


class TrainingStoppedError(RuntimeError):
    pass


def write_training_control(control_path: Path, state: str) -> None:
    control_path.parent.mkdir(parents=True, exist_ok=True)
    control_path.write_text(json.dumps({"state": state}), encoding="utf-8")


def read_training_control(control_path: Path | None) -> str:
    if control_path is None or not control_path.exists():
        return "running"

    try:
        data = json.loads(control_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "running"

    state = str(data.get("state", "running"))
    return state if state in {"running", "paused", "stopped"} else "running"


def check_training_control(control_path: Path | None, poll_seconds: float = 1.0) -> None:
    while True:
        state = read_training_control(control_path)
        if state == "stopped":
            raise TrainingStoppedError("Training stopped by user.")
        if state != "paused":
            return
        time.sleep(poll_seconds)
