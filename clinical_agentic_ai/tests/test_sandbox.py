"""Sandbox safety + execution tests."""
from __future__ import annotations

import pandas as pd
import pytest

from backend.core.sandbox import SandboxViolation, compile_function, run_per_row, static_check


def test_static_check_blocks_import():
    with pytest.raises(SandboxViolation):
        static_check("import os\ndef derive(row): return None")


def test_static_check_blocks_dunders():
    with pytest.raises(SandboxViolation):
        static_check("def derive(row): return row.__class__")


def test_static_check_blocks_open():
    code = "def derive(row):\n    open('/etc/passwd').read()\n    return None"
    with pytest.raises(SandboxViolation):
        static_check(code)


def test_compile_runs_simple_function():
    fn = compile_function("def derive(row):\n    return row['x'] + 1")
    assert fn({"x": 41}) == 42


def test_run_per_row_returns_values():
    code = """
def derive(row):
    age = to_int(row['age'])
    if age is None:
        return None
    if age < 18:
        return '<18'
    return '>=18'
"""
    df = pd.DataFrame({"age": [12, 30, None, 45]})
    r = run_per_row(code, df)
    assert r.values == ["<18", ">=18", None, ">=18"]
    assert r.row_errors == []


def test_run_per_row_captures_errors():
    code = "def derive(row):\n    return 1 / 0"
    df = pd.DataFrame({"x": [1, 2]})
    r = run_per_row(code, df)
    assert r.values == [None, None]
    assert len(r.row_errors) == 2
