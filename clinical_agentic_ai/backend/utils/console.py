"""Plain-text console narrator.

Sits alongside the structured logger to give a human a scannable view of
what the pipeline is doing in real time. Output goes to stdout via plain
``print`` so it survives every logging configuration and shows up in the
uvicorn terminal as the run progresses.
"""
from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from typing import Iterator


_RUN_START: dict[str, float] = {}
_PHASE_START: dict[str, float] = {}


def _emit(line: str) -> None:
    print(line, flush=True, file=sys.stdout)


def say(message: str) -> None:
    _emit(f"      {message}")


def run_banner(run_id: str, note: str = "") -> None:
    _RUN_START[run_id] = time.perf_counter()
    bar = "=" * 70
    suffix = f" ({note})" if note else ""
    _emit("")
    _emit(bar)
    _emit(f"  RUN {run_id} starting{suffix}")
    _emit(bar)


def run_resumed(run_id: str, from_phase: str) -> None:
    _RUN_START[run_id] = time.perf_counter()
    _emit("")
    _emit(f">>> RUN {run_id} resuming after {from_phase}")


def run_paused(run_id: str, reason: str) -> None:
    _emit("")
    _emit(f"<<< RUN {run_id} PAUSED — {reason}")
    _emit("    Awaiting reviewer decision in the UI (Pending Decisions page).")


def run_completed(run_id: str, status: str, summary: dict | None = None) -> None:
    elapsed = time.perf_counter() - _RUN_START.pop(run_id, time.perf_counter())
    _emit("")
    _emit(f">>> RUN {run_id} {status.upper()} in {elapsed:.1f}s")
    if summary:
        bits = []
        for k in ("derivations_total", "derivations_ok", "blocking_findings"):
            if k in summary:
                bits.append(f"{k}={summary[k]}")
        if bits:
            _emit("    " + ", ".join(bits))
    _emit("")


@contextmanager
def phase(run_id: str, index: int, total: int, name: str) -> Iterator[None]:
    """Print a phase header on entry and a duration line on exit."""
    key = f"{run_id}:{name}"
    _PHASE_START[key] = time.perf_counter()
    _emit(f"[{index}/{total}] {name} ...")
    try:
        yield
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = int((time.perf_counter() - _PHASE_START.pop(key, time.perf_counter())) * 1000)
        _emit(f"        FAILED in {elapsed_ms}ms: {type(exc).__name__}: {exc}")
        raise
    else:
        elapsed_ms = int((time.perf_counter() - _PHASE_START.pop(key, time.perf_counter())) * 1000)
        _emit(f"        done in {elapsed_ms}ms")


def llm_call_start(model: str, purpose: str) -> None:
    _emit(f"      -> LLM call ({purpose}) -> {model}")


def llm_call_done(latency_ms: int, tokens_in: int, tokens_out: int) -> None:
    _emit(
        f"      <- LLM reply in {latency_ms / 1000:.1f}s "
        f"({tokens_in} in / {tokens_out} out tokens)"
    )


def hitl_decision(run_id: str, reviewer: str, action: str, target: str | None) -> None:
    tgt = f" on {target}" if target else ""
    _emit("")
    _emit(f">>> HITL decision applied: {reviewer} -> {action}{tgt} (run {run_id})")
