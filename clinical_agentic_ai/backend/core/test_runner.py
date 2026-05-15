"""Test-case runner — validate generated code against spec-declared examples.

The Phase-0 quality gate. Each derivation in the spec can declare a list of
``test_cases``, e.g.::

    test_cases:
      - input: {age: 12}
        expected: "<18"
      - input: {age: 18}
        expected: "18-64"

Before the executor spends compute running the generated code on a
30,000-row dataset, the test runner exercises it against these examples
first. If any case fails, the derivation is marked failed and routed to
the Refiner — short-circuiting wasted work and giving the Refiner
concrete, structured evidence of WHAT is wrong instead of a generic
"verification failed".

This is the single most effective defence against "the code generator's
output looked plausible but quietly produced wrong numbers."
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.core.sandbox import SandboxViolation, compile_function


@dataclass
class TestCaseResult:
    case_index: int
    input: dict[str, Any]
    expected: Any
    actual: Any
    passed: bool
    error: str | None = None


@dataclass
class TestRunResult:
    target: str
    cases: list[TestCaseResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.cases)

    @property
    def passed_count(self) -> int:
        return sum(1 for c in self.cases if c.passed)

    @property
    def all_passed(self) -> bool:
        return self.total > 0 and self.passed_count == self.total

    @property
    def failed_cases(self) -> list[TestCaseResult]:
        return [c for c in self.cases if not c.passed]


def _values_equal(actual: Any, expected: Any) -> bool:
    """Compare with sensible None and numeric tolerance semantics."""
    if actual is None and expected is None:
        return True
    if actual is None or expected is None:
        return False
    if isinstance(actual, bool) or isinstance(expected, bool):
        return actual == expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        try:
            return abs(float(actual) - float(expected)) < 1e-9
        except (TypeError, ValueError):
            return False
    return actual == expected


def run_test_cases(
    code: str,
    target: str,
    cases: list[dict[str, Any]],
) -> TestRunResult:
    """Compile ``code`` once and run each test case through it.

    A test case is a dict ``{"input": {col: value, ...}, "expected": value}``.
    The runner is tolerant of test cases missing ``input`` or ``expected`` —
    they simply count as failures with a descriptive error.
    """
    result = TestRunResult(target=target)
    if not cases:
        return result

    try:
        fn = compile_function(code)
    except SandboxViolation as exc:
        for i, c in enumerate(cases):
            result.cases.append(TestCaseResult(
                case_index=i,
                input=c.get("input") or {},
                expected=c.get("expected"),
                actual=None,
                passed=False,
                error=f"compile failed: {exc}",
            ))
        return result

    for i, c in enumerate(cases):
        if not isinstance(c, dict) or "input" not in c or "expected" not in c:
            result.cases.append(TestCaseResult(
                case_index=i,
                input=(c.get("input") if isinstance(c, dict) else {}) or {},
                expected=(c.get("expected") if isinstance(c, dict) else None),
                actual=None,
                passed=False,
                error="malformed test case (missing input or expected)",
            ))
            continue
        inp = c["input"] or {}
        expected = c["expected"]
        try:
            actual = fn(dict(inp))
        except Exception as exc:  # noqa: BLE001
            result.cases.append(TestCaseResult(
                case_index=i,
                input=inp,
                expected=expected,
                actual=None,
                passed=False,
                error=f"{type(exc).__name__}: {exc}",
            ))
            continue
        result.cases.append(TestCaseResult(
            case_index=i,
            input=inp,
            expected=expected,
            actual=actual,
            passed=_values_equal(actual, expected),
        ))
    return result
