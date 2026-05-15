"""
CodeGenerator agent — produces one sandboxed ``derive(row)`` function per
declared derivation, in topological order.

The priority chain has two stages, no hardcoded fallback library:

  1. Long-Term Memory hit. If a validated pattern with the same signature
     (target name + sorted sources + tokenised rule text) exists, reuse it.
     Cheap, deterministic, and consistent across studies — a human-edited
     fix in one study automatically benefits the next.
  2. LLM call with a strict JSON contract. The model returns ``code``,
     ``reasoning``, ``confidence``, and ``uncertainty_notes``. The reasoning
     trace is what makes the system genuinely agentic rather than a
     workflow: every code-generation decision carries its own justification
     into the audit trail.

There is no hardcoded rule library here by design. If the LLM is offline
and memory is empty, the derivation is marked ``failed`` with a clear error
and the orchestrator escalates to a manual-code-entry HITL gate. That is
the honest contract — an LLM or a vetted human override is required to
synthesise code for a novel derivation; a fake library masquerading as a
fallback gives false confidence on rules it was never written for.
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

from backend.agents.base import BaseAgent
from backend.core import checkpoint
from backend.core.config import settings
from backend.core.guardrails import check_generated_code
from backend.core.llm_client import LLMError, llm
from backend.core.workflow_state import DerivationRecord, ValidationRecord, WorkflowState
from backend.memory.long_term import LongTermMemory
from backend.utils.hashing import hash_text


# Bump the version when this prompt's output contract changes.
PROMPT_VERSION = "code_generator/v5"  # v5 grounds raw-source values from the dataset


def _sample_source_values(df: pd.DataFrame, col: str, k: int = 12) -> dict[str, Any]:
    """Return a small, deterministic descriptor of a column's observed values.

    The LLM is grounded on what is actually in the dataset rather than what it
    guesses might be there. For categoricals/strings: the top-k distinct values
    with counts. For numerics: min / quantiles / max. For dates: min / max.
    Output is intentionally compact so it does not bloat the prompt.
    """
    if col not in df.columns:
        return {"column": col, "absent": True}
    s = df[col]
    n = int(s.size)
    nulls = int(s.isna().sum())
    info: dict[str, Any] = {"column": col, "rows": n, "nulls": nulls}
    non_null = s.dropna()
    if non_null.empty:
        info["all_null"] = True
        return info
    if pd.api.types.is_numeric_dtype(non_null):
        info["dtype"] = "numeric"
        info["min"] = float(non_null.min())
        info["max"] = float(non_null.max())
        try:
            qs = non_null.quantile([0.25, 0.5, 0.75]).to_dict()
            info["q25"], info["q50"], info["q75"] = (
                float(qs[0.25]), float(qs[0.5]), float(qs[0.75])
            )
        except Exception:  # noqa: BLE001
            pass
    elif pd.api.types.is_datetime64_any_dtype(non_null):
        info["dtype"] = "datetime"
        info["min"] = str(non_null.min())
        info["max"] = str(non_null.max())
    else:
        info["dtype"] = "categorical"
        vc = non_null.astype(str).value_counts().head(k)
        info["distinct"] = int(non_null.astype(str).nunique())
        info["top"] = [
            {"value": str(idx), "count": int(cnt)} for idx, cnt in vc.items()
        ]
    return info


def _format_observed(info: dict[str, Any]) -> str:
    """One-line human-readable rendering of ``_sample_source_values`` output.
    Designed to read naturally inside the LLM prompt without bloating it."""
    if info.get("absent"):
        return "(column not present in dataset)"
    if info.get("all_null"):
        return f"{info['rows']} rows, all null"
    dtype = info.get("dtype", "?")
    nulls = info.get("nulls", 0)
    rows = info.get("rows", 0)
    null_note = f", {nulls}/{rows} null" if nulls else ""
    if dtype == "numeric":
        return (
            f"numeric, min={info.get('min')} max={info.get('max')} "
            f"q25={info.get('q25')} q50={info.get('q50')} q75={info.get('q75')}"
            f"{null_note}"
        )
    if dtype == "datetime":
        return f"datetime, range {info.get('min')} -> {info.get('max')}{null_note}"
    top = info.get("top") or []
    distinct = info.get("distinct", 0)
    rendered = ", ".join(f"{t['value']!r} (n={t['count']})" for t in top)
    suffix = ""
    if distinct > len(top):
        suffix = f" ... and {distinct - len(top)} more distinct value(s)"
    return f"categorical, {distinct} distinct: {rendered}{suffix}{null_note}"


_SYSTEM_PROMPT = """You are a clinical data derivation code generator.

Each prompt provides:
  - target: the name of the column you must compute
  - type: the declared output type (int / float / string / category / bool / date)
  - allowed_values: the closed set, if the type is category / enum
  - rule: natural-language description of the transformation
  - sources: a list of named inputs, each annotated as either
      RAW (a column in the input dataset, available directly)
      or DERIVED (a column computed by an earlier derivation in this same
      spec; populated by the upstream derive function before yours runs).
      RAW sources are accompanied by the OBSERVED VALUE PROFILE — the
      actual distinct values (or numeric range / date range) present in
      this run's dataset. Treat that profile as the SOURCE OF TRUTH for
      what your code must handle: do NOT invent synonyms or alternate
      spellings. If the data says `RECOVERING`, your code branches on
      `RECOVERING` — not `Recovering`, not `RECOVERY`, not `ONGOING`.
  - test_cases: concrete (input, expected) pairs your code MUST satisfy
  - regenerate_hint: optional reviewer feedback when this is a retry

Produce a Python function with EXACTLY this signature:

def derive(row):
    ...
    return value

How sources work at runtime:
  - Both RAW and DERIVED sources are keys of the ``row`` dict. Access them
    the same way: ``row["column_name"]``. The executor populates derived
    values from upstream derive functions before yours runs, in topological
    order.
  - DERIVED source values are the OUTPUT of the upstream function. If the
    upstream type is int, the value is a Python int or None. If float,
    float or None. If category, a string or None. Coerce explicitly with
    ``to_int`` / ``to_float`` before numeric comparisons so a string- or
    object-typed value never sabotages a >= check.
  - Treat None / missing as "skip": return None or the appropriate fallback
    rather than raising.

Sandbox rules (these are NOT suggestions — code that breaks them is
rejected by the static validator or fails at execution):

  Helpers available (call by name):
    isna(x)                   -> True if x is None or NaN
    notna(x)                  -> the opposite
    to_int(x)                 -> int or None
    to_float(x)               -> float or None
    days_between(later, earlier) -> float (later - earlier).days, or None
                                    if either input is missing / unparseable.
                                    NOTE THE ARGUMENT ORDER: the LATER date
                                    comes FIRST. Forgetting this is the most
                                    common bug; use a named call site if it
                                    helps, e.g.:
                                      duration = days_between(
                                          later=row["visit_date"],
                                          earlier=row["treatment_start_date"],
                                      )
    math (sub-module): math.floor, math.ceil, math.sqrt, math.log, math.exp
    Built-ins: abs, min, max, len, round, sum, any, all, str, int, float,
               bool, True, False, None.

  EXPLICITLY FORBIDDEN — these will fail and your output will be all-None:
    - isinstance / hasattr / getattr / setattr  (not in the namespace)
    - row.get(...) or any attribute access on the row dict
      (use row["name"] only; if the key may be missing, check with
       isna or guard the value first)
    - import / from / open / exec / eval / print / try-except blocks
    - assertions, raise statements, anything that uses __dunders__

  Defensive coercion is required:
    - For DERIVED numeric sources, always pass through to_int or to_float
      before comparison. Do NOT compare row["X"] directly with a number.
    - For DERIVED string sources, the value may already be a Python str
      or may be None. Use isna(x) first, then str(x).strip().upper() to
      normalise — but only after the isna guard.
    - Never call .strip(), .upper(), .lower() on a value that might be
      None.

You MUST also self-report:
  - reasoning: 1-3 sentences explaining the structure of your code and how
    you interpreted the rule. Be specific, not generic. Bad: "I wrote a
    function for the derivation". Good: "I coerce the derived numeric
    source via to_int because the row-dict value may be int, float, or
    None; rows where it is None or below the threshold map to the failure
    branch per the rule."
  - confidence: a number between 0.0 and 1.0 — your subjective certainty
    that this code is correct for ALL realistic inputs (not just the test
    cases). Low confidence flags the derivation for mandatory human review.
  - uncertainty_notes: one short sentence describing anything ambiguous or
    risky about your interpretation. Empty string if none.

Reply with strict JSON: {"code": "<full function>", "reasoning": "...",
"confidence": 0.85, "uncertainty_notes": "..."} — NO commentary outside
the JSON.
"""


_CODE_KEY_RE = re.compile(r'"code"\s*:\s*"', re.IGNORECASE)


class CodeGeneratorAgent(BaseAgent):
    name = "code_generator"
    step = "generate"

    def __init__(self, stm, ltm: LongTermMemory) -> None:
        super().__init__(stm)
        self.ltm = ltm

    def run(self, state: WorkflowState) -> WorkflowState:
        with self.step_ctx() as rec:
            derivations = state.spec["normalised_derivations"]
            source_schema: dict[str, str] = state.spec.get("source_schema") or {}
            schema_cols = list(source_schema.keys())
            all_columns = schema_cols + [d["name"] for d in derivations]
            # Map derived target names back to their full spec entry so the
            # prompt can describe each source as RAW (type from schema) or
            # DERIVED (type + rule from the upstream derivation).
            derived_map: dict[str, dict[str, Any]] = {
                d["name"]: d for d in derivations
            }
            # Observed value profiles for every RAW column. Loaded once and
            # cached because every derivation that uses the same source gets
            # the same profile — keeps the grounding consistent and avoids
            # re-reading the dataset per derivation.
            observed_values: dict[str, dict[str, Any]] = {}
            try:
                if state.dataset_path:
                    df_obs = (
                        pd.read_csv(state.dataset_path)
                        if state.dataset_path.endswith(".csv")
                        else pd.read_parquet(state.dataset_path)
                    )
                    for col in schema_cols:
                        observed_values[col] = _sample_source_values(df_obs, col)
            except Exception as exc:  # noqa: BLE001
                self.log.warning("codegen.observed_values_failed", error=str(exc))
            regen_hints = state.spec.get("regenerate_hints") or {}
            generated = 0
            from_memory = 0
            from_llm = 0
            failed = 0

            # Live progress fields. The frontend polls /runs/{id}/state every
            # ~2s and renders ``current_target`` in a single-line status under
            # the run header. The inflight checkpoint below makes these
            # fields visible mid-phase; without it the user would stare at a
            # generic "processing" spinner for the entire codegen pass.
            total_targets = len(derivations)
            state.summary["current_phase"] = "code_generation"
            state.summary["derivations_total"] = total_targets
            state.summary["derivations_done"] = 0

            for idx, d in enumerate(derivations, start=1):
                target = d["name"]
                if target in state.derivations and state.derivations[target].status == "ok":
                    state.summary["derivations_done"] = idx
                    continue  # already done (e.g., resumed run)

                # Publish "currently working on X" before the LLM call so a
                # 5-10s codegen round trip does not look like a stall.
                state.summary["current_target"] = target
                state.summary["current_index"] = idx
                checkpoint.save(
                    state.run_id, "05_generate_inflight", state.to_dict()
                )

                hint = regen_hints.get(target)
                code, generator, confidence, notes, reasoning = self._produce_code(
                    d,
                    hint=hint,
                    source_schema=source_schema,
                    derived_map=derived_map,
                    observed_values=observed_values,
                )

                # No code produced — record failure with a clear validator finding.
                if not code:
                    msg = (
                        "Code generation failed: LLM disabled or returned no "
                        "valid code and no long-term memory hit available."
                    )
                    state.validations.append(ValidationRecord(
                        rule_id="CODEGEN_NO_OUTPUT",
                        target=target,
                        severity="block",
                        passed=False,
                        message=msg,
                    ))
                    self.stm.vals.record(
                        run_id=state.run_id, target=target,
                        rule_id="CODEGEN_NO_OUTPUT",
                        severity="block", passed=False, message=msg,
                    )
                    state.derivations[target] = DerivationRecord(
                        target=target,
                        sources=list(d.get("sources", [])),
                        rule_text=d.get("rule", ""),
                        code="",
                        code_hash="",
                        generator=generator,
                        attempt=1,
                        status="failed",
                        confidence=confidence,
                        uncertainty_notes=notes or "code generation failed",
                        reasoning=reasoning,
                        risk_class=str(d.get("risk_class") or "routine"),
                    )
                    self.stm.derivs.upsert(
                        run_id=state.run_id, target=target,
                        sources=list(d.get("sources", [])),
                        rule_text=d.get("rule", ""), code="", code_hash="",
                        generator=generator, attempt=1, status="failed",
                    )
                    failed += 1
                    continue

                guard = check_generated_code(code, available_columns=all_columns)
                for f in guard.findings:
                    state.validations.append(ValidationRecord(
                        rule_id=f.code, target=target, severity=f.severity,
                        passed=False, message=f.message, detail=f.details,
                    ))
                    self.stm.vals.record(
                        run_id=state.run_id, target=target, rule_id=f.code,
                        severity=f.severity, passed=False, message=f.message, detail=f.details,
                    )

                code_hash = hash_text(code)
                state.derivations[target] = DerivationRecord(
                    target=target,
                    sources=list(d.get("sources", [])),
                    rule_text=d.get("rule", ""),
                    code=code,
                    code_hash=code_hash,
                    generator=generator,
                    attempt=1,
                    status="generated" if guard.ok else "unsafe",
                    confidence=confidence,
                    uncertainty_notes=notes,
                    reasoning=reasoning,
                    risk_class=str(d.get("risk_class") or "routine"),
                )
                self.stm.derivs.upsert(
                    run_id=state.run_id, target=target,
                    sources=list(d.get("sources", [])),
                    rule_text=d.get("rule", ""), code=code, code_hash=code_hash,
                    generator=generator, attempt=1,
                    status="generated" if guard.ok else "unsafe",
                )
                generated += 1
                if generator == "memory":
                    from_memory += 1
                elif generator == "llm":
                    from_llm += 1
                state.summary["derivations_done"] = idx

            # Clear the live-progress fields. The next phase will set its
            # own current_phase; leaving a stale current_target would
            # confuse the frontend indicator.
            state.summary.pop("current_target", None)
            state.summary.pop("current_index", None)

            # Phase 5: count how many generated derivations fell below the
            # confidence threshold — used downstream for early HITL.
            low_conf = sum(
                1 for dx in state.derivations.values()
                if dx.confidence is not None
                and dx.confidence < float(settings.min_confidence_threshold)
                and dx.status in ("generated", "refined")
            )
            rec.detail.update({
                "generated": generated,
                "from_memory": from_memory,
                "from_llm": from_llm,
                "failed_no_output": failed,
                "low_confidence_count": low_conf,
                "confidence_threshold": float(settings.min_confidence_threshold),
            })
        return state

    # -------------------------- helpers --------------------------

    def _produce_code(
        self,
        deriv: dict[str, Any],
        *,
        hint: str | None = None,
        source_schema: dict[str, str] | None = None,
        derived_map: dict[str, dict[str, Any]] | None = None,
        observed_values: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[str, str, float | None, str | None, str | None]:
        """Return ``(code, generator_label, confidence, uncertainty_notes, reasoning)``.

        ``code`` may be an empty string when generation fails entirely. The
        caller is responsible for raising a validator block + escalating.
        ``reasoning`` is the LLM's explanation of why it produced this code,
        stored in the audit trail.
        """
        target = deriv["name"]
        sources = list(deriv.get("sources", []))
        rule = deriv.get("rule", "")

        # 1. LTM. When a pattern hits, surface the LLM's original reasoning
        # captured the first time this signature was promoted. The lookup
        # is keyed on the current PROMPT_VERSION so a prompt bump
        # invalidates stale patterns automatically.
        ltm_hit = self.ltm.lookup(
            target=target, sources=sources, rule_text=rule,
            generator_version=PROMPT_VERSION,
        )
        if ltm_hit is not None:
            self.ltm.reinforce(ltm_hit.id)
            stored_reasoning = (ltm_hit.reasoning or "").strip()
            if stored_reasoning:
                reasoning_text = (
                    f"{stored_reasoning} "
                    f"(reused from memory pattern #{ltm_hit.id}, "
                    f"used {ltm_hit.score:.0f} time(s) before)."
                )
            else:
                reasoning_text = (
                    f"Reused validated pattern (memory id={ltm_hit.id}, "
                    f"score={ltm_hit.score:.2f})."
                )
            return ltm_hit.code, "memory", 1.0, None, reasoning_text

        # 2. LLM
        if not llm.enabled:
            return "", "llm_disabled", None, "LLM is not configured.", None

        # Build an annotated source description. The LLM needs to know which
        # sources are raw columns (available straight from the dataset) and
        # which are derived (produced by an earlier derivation in this spec);
        # otherwise it sometimes treats a DERIVED int as a string or assumes
        # the column is missing.
        source_schema = source_schema or {}
        derived_map = derived_map or {}
        observed_values = observed_values or {}
        source_lines: list[str] = []
        for src in sources:
            if src in source_schema:
                profile = ""
                if src in observed_values:
                    profile = (
                        f"\n      observed in this dataset: "
                        f"{_format_observed(observed_values[src])}"
                    )
                source_lines.append(
                    f"  - {src}: RAW input column, type={source_schema[src]}"
                    f"{profile}"
                )
            elif src in derived_map:
                up = derived_map[src]
                up_rule = " ".join((up.get("rule") or "").split())[:100]
                allowed = up.get("allowed_values")
                allowed_note = (
                    f"\n      upstream produces values in {sorted(set(allowed))}"
                    if allowed else ""
                )
                source_lines.append(
                    f"  - {src}: DERIVED column computed earlier in this spec, "
                    f"type={up.get('type', '?')}; "
                    f"upstream rule: {up_rule or '(no rule text)'}"
                    f"{allowed_note}"
                )
            else:
                source_lines.append(
                    f"  - {src}: unknown origin (treat as nullable, coerce types defensively)"
                )
        sources_block = "sources:\n" + "\n".join(source_lines) + "\n"

        tests = deriv.get("test_cases") or []
        tests_block = ""
        if tests:
            tests_block = (
                "test_cases (your code MUST satisfy each one):\n"
                + "\n".join(
                    f"  - derive({tc.get('input')!r}) should return {tc.get('expected')!r}"
                    for tc in tests
                )
                + "\n"
            )
        hint_block = f"regenerate_hint (reviewer): {hint}\n" if hint else ""

        try:
            resp = llm.complete(
                system=_SYSTEM_PROMPT,
                user=(
                    f"target: {target}\n"
                    f"type: {deriv.get('type')}\n"
                    f"allowed_values: {deriv.get('allowed_values')}\n"
                    f"{sources_block}"
                    f"rule: {rule}\n"
                    f"{tests_block}"
                    f"{hint_block}"
                ),
                expect_json=True,
                purpose=f"{PROMPT_VERSION}:{target}",
            )
            parsed = resp.parsed or {}
            code = (parsed.get("code") or "").strip()
            raw_conf = parsed.get("confidence")
            confidence: float | None = None
            try:
                if raw_conf is not None:
                    confidence = max(0.0, min(1.0, float(raw_conf)))
            except (TypeError, ValueError):
                confidence = None
            notes = (parsed.get("uncertainty_notes") or "").strip() or None
            reasoning = (parsed.get("reasoning") or "").strip() or None
            if code:
                return code, "llm", confidence, notes, reasoning
            return "", "llm", confidence, notes or "LLM returned no code.", reasoning
        except LLMError as exc:
            self.log.warning("codegen.llm_failed", target=target, error=str(exc))
            return "", "llm_error", None, f"LLM error: {exc}", None
