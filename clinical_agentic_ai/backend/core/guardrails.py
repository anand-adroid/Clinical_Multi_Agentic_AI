"""
Input & Output guardrails.

These are the *contractual* checks that sit at the boundary of every agent
and on the system as a whole. They are intentionally distinct from in-domain
validation rules (which live in the Verifier agent) — guardrails enforce
*safety and shape*, the Verifier enforces *correctness*.

The five guardrail families
---------------------------
1. **Input schema guard** — the incoming dataset matches the declared schema
   (columns present, types coercible, no impossible nulls).
2. **Spec guard** — the transformation spec parses, has unique target names,
   resolves all references, and has no obvious sentinel values
   (`TODO`, `???`, `<INSERT>`).
3. **PII guard** — the dataset does not contain obvious PII columns (names,
   emails, SSN-like strings) that should not be present in a derivation
   pipeline. The behaviour here is BLOCK (not redact) and surface to HITL.
4. **Code-safety guard** — wraps `sandbox.static_check`, plus heuristic checks
   against patterns the LLM might emit that are syntactically allowed but
   nonsensical (e.g. references to undeclared columns).
5. **Output guard** — every derived column has the right type, plausible
   cardinality, and an acceptable null rate; otherwise the guard flags it.

Each guard returns a `GuardResult` rather than raising, so the orchestrator
can decide whether to halt, escalate to HITL, or proceed with warnings.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from backend.core.sandbox import SandboxViolation, static_check
from backend.utils.logging_setup import get_logger

log = get_logger("guardrails")


Severity = str  # "info" | "warn" | "block"


@dataclass
class GuardFinding:
    code: str
    message: str
    severity: Severity
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class GuardResult:
    ok: bool
    findings: list[GuardFinding] = field(default_factory=list)

    def block(self, code: str, message: str, **details: Any) -> None:
        self.findings.append(GuardFinding(code, message, "block", details))
        self.ok = False

    def warn(self, code: str, message: str, **details: Any) -> None:
        self.findings.append(GuardFinding(code, message, "warn", details))

    def info(self, code: str, message: str, **details: Any) -> None:
        self.findings.append(GuardFinding(code, message, "info", details))


# --------------------------------------------------------------------------- #
#  PII heuristics                                                             #
# --------------------------------------------------------------------------- #

_PII_COL_HINTS = {
    "name", "first_name", "last_name", "full_name",
    "email", "phone", "ssn", "social_security",
    "address", "postcode", "zip", "dob",
}
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_SSN_RE = re.compile(r"^\d{3}-\d{2}-\d{4}$")


def check_pii(df: pd.DataFrame) -> GuardResult:
    res = GuardResult(ok=True)
    lower_cols = {c.lower(): c for c in df.columns}
    for hint in _PII_COL_HINTS:
        if hint in lower_cols:
            res.block(
                "PII_COLUMN",
                f"Column `{lower_cols[hint]}` looks like PII; not allowed in derivation input.",
                column=lower_cols[hint],
            )
    # Spot-check the first 50 rows of string columns for email/SSN patterns.
    sample = df.head(50)
    for col in sample.select_dtypes(include="object").columns:
        s = sample[col].dropna().astype(str)
        if (s.map(_EMAIL_RE.match).notna().sum() / max(len(s), 1)) > 0.3:
            res.block("PII_EMAIL_VALUES", f"Column `{col}` appears to hold emails.", column=col)
        if (s.map(_SSN_RE.match).notna().sum() / max(len(s), 1)) > 0.3:
            res.block("PII_SSN_VALUES", f"Column `{col}` appears to hold SSNs.", column=col)
    return res


# --------------------------------------------------------------------------- #
#  Input schema guard                                                         #
# --------------------------------------------------------------------------- #


def check_input_schema(df: pd.DataFrame, declared: dict[str, str]) -> GuardResult:
    """`declared`: column -> type ("int" | "float" | "string" | "date" | "category")"""
    res = GuardResult(ok=True)
    missing = [c for c in declared if c not in df.columns]
    if missing:
        res.block("MISSING_COLUMNS", f"Declared source columns missing: {missing}",
                  missing=missing)
        return res

    for col, t in declared.items():
        s = df[col]
        if t == "int":
            coerced = pd.to_numeric(s, errors="coerce")
            if coerced.isna().sum() > s.isna().sum():
                res.warn("INT_COERCION_LOSS",
                         f"Column `{col}` has values that cannot be coerced to int.",
                         column=col)
        elif t == "float":
            coerced = pd.to_numeric(s, errors="coerce")
            if coerced.isna().sum() > s.isna().sum():
                res.warn("FLOAT_COERCION_LOSS",
                         f"Column `{col}` has values that cannot be coerced to float.",
                         column=col)
        elif t == "date":
            coerced = pd.to_datetime(s, errors="coerce")
            if coerced.isna().sum() > s.isna().sum():
                res.warn("DATE_COERCION_LOSS",
                         f"Column `{col}` has values that cannot be parsed as date.",
                         column=col)
        elif t in ("string", "category"):
            pass  # nothing to do
        else:
            res.warn("UNKNOWN_TYPE", f"Unknown declared type `{t}` for `{col}`.", column=col, type=t)
    return res


# --------------------------------------------------------------------------- #
#  Spec guard                                                                 #
# --------------------------------------------------------------------------- #

_SUSPICIOUS_TOKENS = ("TODO", "???", "<INSERT", "FIXME", "TBD")


def check_spec(spec: dict[str, Any]) -> GuardResult:
    res = GuardResult(ok=True)
    if "derivations" not in spec or not isinstance(spec["derivations"], list):
        res.block("SPEC_NO_DERIVATIONS", "Spec is missing a `derivations` list.")
        return res
    seen: set[str] = set()
    for i, d in enumerate(spec["derivations"]):
        name = d.get("name")
        if not name:
            res.block("SPEC_MISSING_NAME", f"Derivation #{i} has no `name`.", index=i)
            continue
        if name in seen:
            res.block("SPEC_DUPLICATE_NAME", f"Duplicate derivation name `{name}`.", name=name)
        seen.add(name)
        rule_text = (d.get("rule") or "") + " " + (d.get("description") or "")
        for tok in _SUSPICIOUS_TOKENS:
            if tok.lower() in rule_text.lower():
                res.warn("SPEC_AMBIGUOUS",
                         f"Derivation `{name}` contains placeholder token `{tok}`.",
                         name=name, token=tok)
    return res


# --------------------------------------------------------------------------- #
#  Code-safety guard                                                          #
# --------------------------------------------------------------------------- #


def check_generated_code(code: str, available_columns: list[str]) -> GuardResult:
    res = GuardResult(ok=True)
    try:
        static_check(code)
    except SandboxViolation as exc:
        res.block("CODE_UNSAFE", str(exc))
        return res
    # Quick textual sanity: at least one referenced column should exist.
    referenced = set(re.findall(r"row\[['\"]([^'\"]+)['\"]\]", code))
    unknown = referenced - set(available_columns)
    if unknown:
        res.block(
            "CODE_UNKNOWN_COLUMNS",
            f"Generated code references unknown columns: {sorted(unknown)}",
            unknown=sorted(unknown),
        )
    return res


# --------------------------------------------------------------------------- #
#  Output guard                                                               #
# --------------------------------------------------------------------------- #


def check_output_column(
    name: str,
    values: list[Any],
    *,
    expected_type: str | None = None,
    allowed_values: list[Any] | None = None,
    max_null_rate: float = 0.5,
) -> GuardResult:
    res = GuardResult(ok=True)
    n = len(values)
    nulls = sum(1 for v in values if v is None or (isinstance(v, float) and pd.isna(v)))
    null_rate = nulls / n if n else 1.0
    if null_rate > max_null_rate:
        # Two severity tiers. A small overshoot (e.g. 7% null when the
        # spec allows 5%) is real-world acceptable and only worth a warn;
        # the row-level errors are already recorded on the derivation.
        # A massive overshoot — more than double the declared cap, or
        # everything null when the spec asked for none — is almost
        # always a code bug. Escalate to BLOCK so the Refiner gets a
        # chance to fix it before the output goes downstream.
        # Two severity tiers. A small overshoot (e.g. 7% null when the spec
        # allows 5%) is real-world acceptable and only worth a warn — the
        # row-level errors are already recorded on the derivation. A
        # *significant* overshoot is almost always a code bug: a mapping
        # missed a value, a type cast silently dropped rows, the LLM
        # invented an enum that does not exist in the data. Escalate to
        # BLOCK so the Refiner gets a chance to fix it.
        #
        # Trigger when *either*:
        #   - max_null_rate == 0 and any null leaks through, OR
        #   - null_rate > 1.5x cap, with an absolute floor of cap+5% so a
        #     5% cap blocks at 10% (not 7.5%).
        is_egregious = (
            (max_null_rate == 0 and null_rate > 0)
            or null_rate > min(0.5, max(1.5 * max_null_rate, max_null_rate + 0.05))
        )
        if is_egregious:
            res.block(
                "OUTPUT_HIGH_NULL_RATE",
                f"`{name}` is {null_rate:.0%} null (threshold {max_null_rate:.0%}). "
                "Likely a code bug — escalating to Refiner.",
                column=name, null_rate=null_rate,
            )
        else:
            res.warn(
                "OUTPUT_HIGH_NULL_RATE",
                f"`{name}` is {null_rate:.0%} null (threshold {max_null_rate:.0%}).",
                column=name, null_rate=null_rate,
            )
    non_null = [v for v in values if v is not None and not (isinstance(v, float) and pd.isna(v))]
    if allowed_values is not None and non_null:
        bad = sorted({v for v in non_null if v not in set(allowed_values)})
        if bad:
            res.block(
                "OUTPUT_OUT_OF_DOMAIN",
                f"`{name}` has values outside the allowed set: {bad[:5]}",
                column=name, out_of_domain=bad,
            )
    if expected_type and non_null:
        ok = all(_is_of_type(v, expected_type) for v in non_null)
        if not ok:
            res.warn(
                "OUTPUT_TYPE_MISMATCH",
                f"`{name}` contains values inconsistent with type `{expected_type}`.",
                column=name, expected=expected_type,
            )
    return res


def _is_of_type(v: Any, t: str) -> bool:
    if t == "int":
        return isinstance(v, int) and not isinstance(v, bool)
    if t == "float":
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    if t == "string":
        return isinstance(v, str)
    if t == "bool":
        return isinstance(v, bool)
    if t in ("category", "enum"):
        return isinstance(v, (str, int))
    return True
