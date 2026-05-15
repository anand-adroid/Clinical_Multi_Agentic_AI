"""
BaseAgent — the common contract every agent in this pipeline obeys.

An "agent" here is a deterministic, side-effect-aware module with a single
public method ``run(state) -> state``. It is not a free-floating LLM call;
it is a typed Python class with:

  - a ``name`` (used for logging, audit, dashboards),
  - pre and post invariants enforced by the orchestrator,
  - a checkpoint written at the end of every step,
  - structured logs (``begin_step`` / ``end_step``).

I use the word "agent" deliberately. Each module has autonomy in HOW it
satisfies its goal — call an LLM, look up LTM, escalate to HITL, retry
with a different prompt — but the public INTERFACE is rigid. That split is
what makes the orchestration predictable in a regulated setting while
keeping the per-agent intelligence agentic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Iterator

from backend.core.workflow_state import StepRecord, WorkflowState
from backend.memory.short_term import ShortTermMemory
from backend.utils.logging_setup import get_logger


class BaseAgent(ABC):
    name: str = "agent"
    step: str = "step"

    def __init__(self, stm: ShortTermMemory) -> None:
        self.stm = stm
        self.log = get_logger(f"agent.{self.name}")

    @contextmanager
    def step_ctx(self, step: str | None = None) -> Iterator[StepRecord]:
        step_id = step or self.step
        rec = self.stm.begin_step(self.name, step_id)
        try:
            yield rec
            self.stm.end_step(rec, status="ok")
        except Exception as exc:  # noqa: BLE001
            self.stm.end_step(rec, status="failed", detail={"error": f"{type(exc).__name__}: {exc}"})
            self.log.error("agent.failed", agent=self.name, step=step_id, error=str(exc))
            raise
        finally:
            self.stm.checkpoint(f"{rec.agent}.{rec.step}")

    @abstractmethod
    def run(self, state: WorkflowState) -> WorkflowState:  # pragma: no cover
        raise NotImplementedError
