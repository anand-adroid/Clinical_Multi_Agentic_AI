"""
Safe execution sandbox for agent-generated derivation code.

Threat model
------------
The code generator (LLM or memory) emits Python that may be wrong,
unintentionally harmful (``os.system("rm -rf /")``), or even adversarial
if the spec author or upstream LLM has been compromised. I treat every
generated snippet as untrusted.

Defences (four layers)
----------------------
1. AST whitelist. Before execution the AST is walked and any node outside
   a small whitelist (literals, arithmetic, comparisons, calls into a
   pre-approved namespace) raises ``SandboxViolation``. No ``import``, no
   ``open``, no ``exec``, no dunder attribute access.
2. Namespace pinning. The snippet runs with ``__builtins__`` stripped down
   to a hand-picked dict; the only callables available are the typed
   helpers (``to_int``, ``days_between``, etc.) bound in explicitly.
3. Row-level contract. Each derivation defines a single pure function
   ``derive(row: dict) -> Any``. The executor invokes it per-row,
   capturing exceptions per-row so a bad input never poisons the column.
4. Time bound. A wall-clock budget per derivation; on overrun the run is
   aborted cleanly (no ``thread.kill`` — a watchdog flag is inspected
   between rows).

Conservative by design. A real production deployment would still add
container or seccomp isolation around this; the AST-level defence here
gives a clean explanation of safety properties to a security reviewer
without depending on infrastructure.
"""
from __future__ import annotations

import ast
import math
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Callable

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
#  Whitelisted AST nodes                                                      #
# --------------------------------------------------------------------------- #

_ALLOWED_NODES: tuple[type, ...] = (
    ast.Module,
    ast.Expression,
    ast.FunctionDef,
    ast.Return,
    ast.If,
    ast.IfExp,
    ast.BoolOp,
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Subscript,
    ast.Index,  # py<3.9 fallback
    ast.Slice,
    ast.Tuple,
    ast.List,
    ast.Dict,
    ast.Set,
    ast.Attribute,  # constrained further by the whitelist below
    ast.arguments,
    ast.arg,
    ast.keyword,
    ast.And,
    ast.Or,
    ast.Not,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Is,
    ast.IsNot,
    ast.Assign,
    ast.AugAssign,
    ast.Store,
    ast.For,
    ast.Pass,
    ast.Break,
    ast.Continue,
    ast.While,
    ast.Expr,
    ast.JoinedStr,
    ast.FormattedValue,
)

_ALLOWED_ATTRS = {
    "days", "year", "month", "day", "lower", "upper", "strip", "startswith",
    "endswith", "isna", "notna", "value", "date",
}


class SandboxViolation(RuntimeError):
    """Raised when generated code uses a disallowed construct."""


def static_check(code: str, func_name: str = "derive") -> None:
    """Walk the AST and reject anything outside the whitelist."""
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise SandboxViolation(f"Syntax error: {exc}") from exc

    has_func = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            has_func = True
        if not isinstance(node, _ALLOWED_NODES):
            raise SandboxViolation(
                f"Disallowed AST node: {type(node).__name__}"
            )
        if isinstance(node, ast.Attribute) and node.attr not in _ALLOWED_ATTRS:
            raise SandboxViolation(f"Disallowed attribute access: .{node.attr}")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise SandboxViolation(f"Disallowed dunder name: {node.id}")
        if isinstance(node, ast.Call):
            # Only allow calls to whitelisted helpers (resolved at runtime by
            # namespace), and never via attributes (e.g. no `os.system`).
            if isinstance(node.func, ast.Attribute) and node.func.attr not in _ALLOWED_ATTRS:
                raise SandboxViolation(
                    f"Disallowed method call: .{node.func.attr}"
                )

    if not has_func:
        raise SandboxViolation(f"Missing required function `{func_name}`")


# --------------------------------------------------------------------------- #
#  Sandbox namespace                                                          #
# --------------------------------------------------------------------------- #


def _safe_isna(x: Any) -> bool:
    try:
        return pd.isna(x)
    except Exception:
        return x is None


def _safe_days_between(later: Any, earlier: Any) -> float | None:
    if _safe_isna(later) or _safe_isna(earlier):
        return None
    a = pd.to_datetime(later, errors="coerce")
    b = pd.to_datetime(earlier, errors="coerce")
    if pd.isna(a) or pd.isna(b):
        return None
    return float((a - b).days)


def _safe_to_int(x: Any) -> int | None:
    if _safe_isna(x):
        return None
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _safe_to_float(x: Any) -> float | None:
    if _safe_isna(x):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


_BUILTINS = {
    "abs": abs,
    "min": min,
    "max": max,
    "len": len,
    "round": round,
    "sum": sum,
    "any": any,
    "all": all,
    "int": _safe_to_int,
    "float": _safe_to_float,
    "str": str,
    "bool": bool,
    "True": True,
    "False": False,
    "None": None,
}

_HELPERS = {
    "isna": _safe_isna,
    "notna": lambda x: not _safe_isna(x),
    "days_between": _safe_days_between,
    "to_int": _safe_to_int,
    "to_float": _safe_to_float,
    "math": _RestrictedMath() if False else None,  # see below
}

# A pared-down math module (no `eval`, `exec`, etc.).
class _RestrictedMath:
    __slots__ = ()
    floor = staticmethod(math.floor)
    ceil = staticmethod(math.ceil)
    sqrt = staticmethod(math.sqrt)
    log = staticmethod(math.log)
    exp = staticmethod(math.exp)


_HELPERS["math"] = _RestrictedMath()


# --------------------------------------------------------------------------- #
#  Per-row execution                                                          #
# --------------------------------------------------------------------------- #


@dataclass
class RowError:
    row_index: int
    error: str


@dataclass
class ExecutionResult:
    values: list[Any]
    row_errors: list[RowError] = field(default_factory=list)
    duration_ms: int = 0


def compile_function(code: str, func_name: str = "derive") -> Callable[[dict], Any]:
    """Compile a snippet that has already passed `static_check`."""
    static_check(code, func_name)
    namespace: dict[str, Any] = {"__builtins__": _BUILTINS, **_HELPERS}
    exec(compile(code, filename="<agent-derivation>", mode="exec"), namespace)  # noqa: S102
    fn = namespace.get(func_name)
    if not callable(fn):
        raise SandboxViolation(f"`{func_name}` is not callable after compile")
    return fn


def run_per_row(
    code: str,
    df: pd.DataFrame,
    *,
    max_seconds: float = 30.0,
    func_name: str = "derive",
) -> ExecutionResult:
    fn = compile_function(code, func_name=func_name)
    out: list[Any] = []
    errors: list[RowError] = []
    t0 = time.perf_counter()
    deadline = t0 + max_seconds

    rows = df.to_dict(orient="records")
    for idx, row in enumerate(rows):
        if time.perf_counter() > deadline:
            raise TimeoutError(
                f"Derivation exceeded {max_seconds:.1f}s after row {idx}/{len(rows)}"
            )
        try:
            v = fn(row)
            # Type-normalise: numpy/pandas scalars -> python primitives
            if isinstance(v, (np.generic,)):
                v = v.item()
            if isinstance(v, (pd.Timestamp, datetime, date)):
                v = v.isoformat() if hasattr(v, "isoformat") else str(v)
            out.append(v)
        except Exception as exc:  # noqa: BLE001
            errors.append(RowError(row_index=idx, error=f"{type(exc).__name__}: {exc}"))
            out.append(None)
    duration = int((time.perf_counter() - t0) * 1000)
    return ExecutionResult(values=out, row_errors=errors, duration_ms=duration)
