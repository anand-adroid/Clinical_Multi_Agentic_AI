"""
Checkpointing.

Every agent step writes a checkpoint blob to disk *before* it returns. If the
process crashes mid-run, the orchestrator can read the latest checkpoint and
resume from there. Two practical consequences:

1. Long workflows (a real clinical study might have hundreds of derivations)
   are robust to transient failures.
2. HITL pauses are just checkpoints — when a reviewer takes the night to make
   a decision, the rest of the system is free to do other work.

Format: one JSON file per (run_id, step_id). Atomic write via tmp+rename.
The workflow state object is what gets serialised; see `workflow_state.py`.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from backend.core.config import settings
from backend.utils.logging_setup import get_logger

log = get_logger("checkpoint")


def _run_dir(run_id: str) -> Path:
    p = Path(settings.checkpoint_dir) / run_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def save(run_id: str, step_id: str, state: dict[str, Any]) -> Path:
    """Atomic write — never leaves a half-written file on disk."""
    p = _run_dir(run_id) / f"{step_id}.json"
    fd, tmp = tempfile.mkstemp(prefix=".ckpt-", dir=p.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, default=str, indent=2)
        os.replace(tmp, p)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    log.info("checkpoint.saved", run_id=run_id, step=step_id, path=str(p))
    return p


def load(run_id: str, step_id: str) -> dict[str, Any] | None:
    p = _run_dir(run_id) / f"{step_id}.json"
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def latest(run_id: str) -> tuple[str, dict[str, Any]] | None:
    p = _run_dir(run_id)
    files = sorted(p.glob("*.json"), key=lambda x: x.stat().st_mtime)
    if not files:
        return None
    last = files[-1]
    with last.open("r", encoding="utf-8") as f:
        return last.stem, json.load(f)


def list_checkpoints(run_id: str) -> list[str]:
    p = _run_dir(run_id)
    return sorted(x.stem for x in p.glob("*.json"))
