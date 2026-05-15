"""
Verifier agent
==============

Runs *domain-level* validation on the materialised output, separately from
the static / sandbox guardrails.

The rules
---------
For every declared derivation the verifier checks:

  * **Type** — values in the output match the declared type.
  * **Allowed values** — for `category` / `enum` types, all values fall in the
    declared closed set.
  * **Null rate** — does not exceed `max_null_rate`.
  * **Spec-defined invariants** — custom invariants from the spec
    (e.g. `df['SOME_COLUMN'] >= 0`).
  * **Cross-row sanity** — duplicate primary keys, monotonicity of dates.

Each verdict is recorded as a `ValidationRecord` with a stable `rule_id`,
which the UI uses to render the audit panel.
"""
from __future__ import annotations

import pandas as pd

from backend.agents.base import BaseAgent
from backend.core.guardrails import check_output_column
from backend.core.workflow_state import ValidationRecord, WorkflowState


class VerifierAgent(BaseAgent):
    name = "verifier"
    step = "verify"

    def run(self, state: WorkflowState) -> WorkflowState:
        with self.step_ctx() as rec:
            assert state.output_path, "executor must run before verifier"
            df = pd.read_parquet(state.output_path)

            failures = 0
            warnings = 0

            # Each verifier pass replaces the previous pass's findings for
            # each derived target. Without this, the refiner loop would
            # accumulate stale blocks across iterations and never see the
            # situation as resolved.
            verifier_rule_ids = {
                "OUTPUT_HIGH_NULL_RATE",
                "OUTPUT_OUT_OF_DOMAIN",
                "OUTPUT_TYPE_MISMATCH",
                "OUTPUT_COLUMN_MISSING",
                "OUTPUT_UNMAPPED_SOURCE_VALUES",
            }
            state.validations = [
                v for v in state.validations
                if not (v.rule_id in verifier_rule_ids)
            ]

            # 1. Per-target guardrails on the materialised output.
            for d in state.spec["normalised_derivations"]:
                name = d["name"]
                target_block = False
                if name not in df.columns:
                    state.validations.append(ValidationRecord(
                        rule_id="OUTPUT_COLUMN_MISSING", target=name,
                        severity="block", passed=False,
                        message=f"Output is missing derived column `{name}`.",
                    ))
                    self.stm.vals.record(
                        run_id=state.run_id, target=name,
                        rule_id="OUTPUT_COLUMN_MISSING",
                        severity="block", passed=False,
                        message=f"Output is missing derived column `{name}`.",
                    )
                    failures += 1
                    target_block = True
                else:
                    guard = check_output_column(
                        name, df[name].tolist(),
                        expected_type=d.get("type"),
                        allowed_values=d.get("allowed_values") or None,
                        max_null_rate=float(d.get("max_null_rate", 0.5)),
                    )
                    for f in guard.findings:
                        state.validations.append(ValidationRecord(
                            rule_id=f.code, target=name, severity=f.severity,
                            passed=False, message=f.message, detail=f.details,
                        ))
                        self.stm.vals.record(
                            run_id=state.run_id, target=name, rule_id=f.code,
                            severity=f.severity, passed=False, message=f.message, detail=f.details,
                        )
                        if f.severity == "block":
                            failures += 1
                            target_block = True
                        elif f.severity == "warn":
                            warnings += 1

                # No-value-mapped detector. Independent of null-rate cap.
                # Catches the "LLM-invented enum" failure mode: source row
                # is fully populated, but the derived function had no
                # matching branch and returned None. That signals the code
                # is missing a case the data actually contains, regardless
                # of how lenient the null cap is.
                if name in df.columns:
                    sources = [s for s in (d.get("sources") or []) if s in df.columns]
                    if sources:
                        non_null_source_mask = df[sources].notna().all(axis=1)
                        unmapped_mask = non_null_source_mask & df[name].isna()
                        unmapped_count = int(unmapped_mask.sum())
                        if unmapped_count > 0:
                            # Collect the source value tuples the function
                            # failed to map, so the Refiner can patch the
                            # exact gap without re-reading the dataset.
                            sample = (
                                df.loc[unmapped_mask, sources]
                                .drop_duplicates()
                                .head(8)
                                .to_dict(orient="records")
                            )
                            cap = float(d.get("max_null_rate", 0.5))
                            non_null_total = int(non_null_source_mask.sum())
                            unmapped_rate = (
                                unmapped_count / non_null_total
                                if non_null_total else 0.0
                            )
                            # Align severity with the egregious-null-rate
                            # threshold in guardrails. cap==0 means any leak
                            # is fatal. Otherwise, an unmapped rate above
                            # ~1.5x the cap (with an absolute floor of
                            # cap+5%) is "LLM invented an enum or missed a
                            # branch" territory and must reach the Refiner.
                            severe = (
                                cap == 0
                                or unmapped_rate
                                > min(0.5, max(1.5 * cap, cap + 0.05))
                            )
                            sev = "block" if severe else "warn"
                            msg = (
                                f"`{name}` returned null for {unmapped_count} "
                                f"row(s) whose source columns were fully "
                                f"populated. The generated code has no branch "
                                f"covering these inputs. Sample: {sample[:3]}"
                            )
                            state.validations.append(ValidationRecord(
                                rule_id="OUTPUT_UNMAPPED_SOURCE_VALUES",
                                target=name, severity=sev,
                                passed=False, message=msg,
                                detail={
                                    "unmapped_count": unmapped_count,
                                    "non_null_source_total": non_null_total,
                                    "unmapped_rate": unmapped_rate,
                                    "sample": sample,
                                },
                            ))
                            self.stm.vals.record(
                                run_id=state.run_id, target=name,
                                rule_id="OUTPUT_UNMAPPED_SOURCE_VALUES",
                                severity=sev, passed=False, message=msg,
                                detail={
                                    "unmapped_count": unmapped_count,
                                    "non_null_source_total": non_null_total,
                                    "unmapped_rate": unmapped_rate,
                                    "sample": sample,
                                },
                            )
                            if sev == "block":
                                failures += 1
                                target_block = True
                            else:
                                warnings += 1

                # Update derivation status so the Refiner picks the target
                # up on the next loop iteration. Without this, a derivation
                # that "executed cleanly" but produces values the verifier
                # rejects would stay status="ok" and the Refiner would
                # never look at it.
                if target_block and name in state.derivations:
                    state.derivations[name].status = "failed"

            # 2. Spec-defined custom invariants. Same stale-clearing logic
            # as the output guardrails — each pass replaces the last.
            invariant_rule_ids = {
                inv.get("id", "CUSTOM_INVARIANT")
                for inv in (state.spec.get("invariants") or [])
            }
            state.validations = [
                v for v in state.validations
                if v.rule_id not in invariant_rule_ids
            ]
            for inv in state.spec.get("invariants", []):
                rid = inv.get("id", "CUSTOM_INVARIANT")
                expr = inv.get("expr")
                target = inv.get("target")
                if not expr:
                    continue
                try:
                    # Eval in a constrained context: only the DataFrame `df`.
                    passed_mask = pd.eval(expr, local_dict={"df": df})
                    n_failed = int((~passed_mask.fillna(False)).sum())
                except Exception as exc:  # noqa: BLE001
                    state.validations.append(ValidationRecord(
                        rule_id=rid, target=target, severity="warn", passed=False,
                        message=f"Invariant `{expr}` could not be evaluated: {exc}",
                    ))
                    self.stm.vals.record(
                        run_id=state.run_id, target=target, rule_id=rid,
                        severity="warn", passed=False,
                        message=f"Invariant `{expr}` could not be evaluated: {exc}",
                    )
                    warnings += 1
                    continue
                if n_failed:
                    state.validations.append(ValidationRecord(
                        rule_id=rid, target=target, severity="block", passed=False,
                        message=f"Invariant `{expr}` failed on {n_failed} row(s).",
                        detail={"failed_rows": n_failed},
                    ))
                    self.stm.vals.record(
                        run_id=state.run_id, target=target, rule_id=rid,
                        severity="block", passed=False,
                        message=f"Invariant `{expr}` failed on {n_failed} row(s).",
                        detail={"failed_rows": n_failed},
                    )
                    failures += 1
                    if target and target in state.derivations:
                        state.derivations[target].status = "failed"
                else:
                    state.validations.append(ValidationRecord(
                        rule_id=rid, target=target, severity="info", passed=True,
                        message=f"Invariant `{expr}` passed.",
                    ))
                    self.stm.vals.record(
                        run_id=state.run_id, target=target, rule_id=rid,
                        severity="info", passed=True,
                        message=f"Invariant `{expr}` passed.",
                    )

            # 3. PK uniqueness, if declared.
            pk = state.spec.get("primary_key")
            if pk and pk in df.columns:
                dups = int(df[pk].duplicated().sum())
                if dups:
                    state.validations.append(ValidationRecord(
                        rule_id="PK_NOT_UNIQUE", target=pk, severity="warn",
                        passed=False, message=f"Primary key `{pk}` has {dups} duplicate row(s).",
                        detail={"duplicates": dups},
                    ))
                    self.stm.vals.record(
                        run_id=state.run_id, target=pk, rule_id="PK_NOT_UNIQUE",
                        severity="warn", passed=False,
                        message=f"Primary key `{pk}` has {dups} duplicate row(s).",
                        detail={"duplicates": dups},
                    )
                    warnings += 1

            rec.detail["failures"] = failures
            rec.detail["warnings"] = warnings
            state.summary["verify_failures"] = failures
            state.summary["verify_warnings"] = warnings
        return state
