"""
Executor agent
==============

Runs each derivation against the dataset in topological order. Produces:

  - the materialised derived columns (parquet under
    ``storage/runs/<run_id>/output.parquet``),
  - per-row error capture (a row that throws records the error and yields
    ``None`` for that cell; one bad row never poisons a column),
  - per-target lineage entries the audit report consumes.

Single-threaded by default — a regulator can replay events in a stable
order, which makes the audit log easy to read. The DAG happens to allow
parallel execution of independent peers at the same topological level
(everything with the same in-degree-zero history). I gated that behind
``settings.parallel_executor`` so the deterministic path stays the default
and the parallel path is opt-in for large datasets.
"""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

from backend.agents.base import BaseAgent
from backend.core.config import settings
from backend.core.sandbox import run_per_row
from backend.core.workflow_state import ValidationRecord, WorkflowState
from backend.utils.hashing import hash_dataframe


def _topo_levels(topo_order: list[str], dag: dict[str, list[str]]) -> list[list[str]]:
    """Group the topological order into levels where every target in a level
    depends only on raw columns or targets from earlier levels. Targets
    inside the same level are independent of each other and safe to run in
    parallel.

    Stable within a level — alphabetical — so the resulting execution order
    is deterministic for any given DAG.
    """
    level_of: dict[str, int] = {}
    for tgt in topo_order:
        deps = [s for s in dag.get(tgt, []) if s in level_of]
        level_of[tgt] = (max(level_of[d] for d in deps) + 1) if deps else 0
    by_level: dict[int, list[str]] = defaultdict(list)
    for tgt, lvl in level_of.items():
        by_level[lvl].append(tgt)
    return [sorted(by_level[k]) for k in sorted(by_level.keys())]


class ExecutorAgent(BaseAgent):
    name = "executor"
    step = "execute"

    def run(self, state: WorkflowState) -> WorkflowState:
        with self.step_ctx() as rec:
            df = (
                pd.read_parquet(state.dataset_path)
                if state.dataset_path.endswith(".parquet")
                else pd.read_csv(state.dataset_path)
            )
            work = df.copy()

            if settings.parallel_executor:
                self._run_parallel(state, work)
                rec.detail["mode"] = "parallel"
                rec.detail["workers"] = int(settings.parallel_executor_workers)
            else:
                self._run_serial(state, work)
                rec.detail["mode"] = "serial"

            out_dir = Path(settings.run_artifact_dir) / state.run_id
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "output.parquet"
            work.to_parquet(out_path, index=False)
            csv_path = out_dir / "output.csv"
            work.to_csv(csv_path, index=False)
            state.output_path = str(out_path)
            rec.outputs_hash = hash_dataframe(work)
            rec.detail["rows"] = len(work)
            rec.detail["columns"] = list(work.columns)
        return state

    # ------------------------------------------------------------------ paths
    def _run_serial(self, state: WorkflowState, work: pd.DataFrame) -> None:
        for target in state.topo_order:
            self._run_one(state, target, work, write_into=work)

    def _run_parallel(self, state: WorkflowState, work: pd.DataFrame) -> None:
        """Run one DAG level at a time; within a level, fan derivations out
        to worker threads. Targets from a level only become readable to the
        next level once all peers in the current level have finished — that
        preserves the dependency contract while extracting wall-clock from
        independent derivations."""
        levels = _topo_levels(state.topo_order, state.dag or {})
        workers = max(1, int(settings.parallel_executor_workers))
        for level in levels:
            results: dict[str, pd.Series] = {}
            with ThreadPoolExecutor(max_workers=min(workers, len(level))) as pool:
                futures = {
                    pool.submit(self._run_one_isolated, state, tgt, work): tgt
                    for tgt in level
                }
                for fut in futures:
                    tgt = futures[fut]
                    series = fut.result()
                    if series is not None:
                        results[tgt] = series
            for tgt, series in results.items():
                work[tgt] = series

    # --------------------------------------------------------- per-target work
    def _run_one(
        self,
        state: WorkflowState,
        target: str,
        readable: pd.DataFrame,
        *,
        write_into: pd.DataFrame,
    ) -> None:
        d = state.derivations[target]
        if d.status == "unsafe":
            return  # leave it for the refiner / HITL
        try:
            result = run_per_row(
                d.code, readable,
                max_seconds=settings.max_runtime_seconds,
            )
        except Exception as exc:  # noqa: BLE001
            self._record_target_failure(state, target, exc)
            return
        write_into[target] = result.values
        self._record_target_success(state, target, result)

    def _run_one_isolated(
        self,
        state: WorkflowState,
        target: str,
        readable: pd.DataFrame,
    ) -> pd.Series | None:
        """Parallel-mode worker: returns the column as a Series so the caller
        can write it back into the shared frame under the main thread."""
        d = state.derivations[target]
        if d.status == "unsafe":
            return None
        try:
            result = run_per_row(
                d.code, readable,
                max_seconds=settings.max_runtime_seconds,
            )
        except Exception as exc:  # noqa: BLE001
            self._record_target_failure(state, target, exc)
            return None
        self._record_target_success(state, target, result)
        return pd.Series(result.values, index=readable.index)

    def _record_target_success(self, state: WorkflowState, target: str, result: Any) -> None:
        d = state.derivations[target]
        d.status = "ok"
        d.null_count = sum(1 for v in result.values if v is None)
        d.row_errors = [
            {"row_index": e.row_index, "error": e.error}
            for e in result.row_errors[:50]
        ]
        for re_ in d.row_errors[:5]:
            # I surface only the first few row-level errors per target as
            # warnings; the full list lives on the derivation record itself.
            state.validations.append(ValidationRecord(
                rule_id="ROW_LEVEL_EXEC_ERROR",
                target=target, severity="warn", passed=False,
                message=re_["error"], detail={"row_index": re_["row_index"]},
            ))
            self.stm.vals.record(
                run_id=state.run_id, target=target,
                rule_id="ROW_LEVEL_EXEC_ERROR",
                severity="warn", passed=False,
                message=re_["error"], detail={"row_index": re_["row_index"]},
            )

    def _record_target_failure(self, state: WorkflowState, target: str, exc: Exception) -> None:
        d = state.derivations[target]
        d.status = "failed"
        msg = f"{type(exc).__name__}: {exc}"
        state.validations.append(ValidationRecord(
            rule_id="EXEC_FAILED", target=target, severity="block",
            passed=False, message=msg,
        ))
        self.stm.vals.record(
            run_id=state.run_id, target=target, rule_id="EXEC_FAILED",
            severity="block", passed=False, message=msg,
        )
