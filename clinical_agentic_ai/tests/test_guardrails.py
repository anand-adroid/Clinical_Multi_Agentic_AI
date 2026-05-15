"""Guardrail unit tests."""
from __future__ import annotations

import pandas as pd

from backend.core.guardrails import (
    check_generated_code,
    check_input_schema,
    check_output_column,
    check_pii,
    check_spec,
)


def test_check_pii_blocks_email_column():
    df = pd.DataFrame({"email": ["a@b.com", "c@d.org"], "age": [10, 20]})
    r = check_pii(df)
    assert not r.ok
    assert any(f.code == "PII_COLUMN" for f in r.findings)


def test_check_pii_blocks_email_values():
    df = pd.DataFrame({"contact": ["a@b.com", "x@y.org", "z@w.net"]})
    r = check_pii(df)
    assert not r.ok


def test_check_input_schema_missing():
    r = check_input_schema(pd.DataFrame({"a": [1]}), {"a": "int", "b": "int"})
    assert not r.ok
    assert r.findings[0].code == "MISSING_COLUMNS"


def test_check_spec_duplicate_names():
    spec = {"derivations": [{"name": "X"}, {"name": "X"}]}
    r = check_spec(spec)
    assert any(f.code == "SPEC_DUPLICATE_NAME" for f in r.findings)


def test_check_generated_code_unsafe():
    code = "import os\ndef derive(row):\n    os.system('ls')"
    r = check_generated_code(code, available_columns=["age"])
    assert not r.ok


def test_check_generated_code_unknown_column():
    code = "def derive(row):\n    return row['nope']"
    r = check_generated_code(code, available_columns=["age"])
    assert not r.ok
    assert any(f.code == "CODE_UNKNOWN_COLUMNS" for f in r.findings)


def test_check_output_column_out_of_domain():
    r = check_output_column("X", ["A", "B", "C"], allowed_values=["A", "B"])
    assert not r.ok


def test_check_output_column_null_rate_warn():
    r = check_output_column("X", [None] * 6 + ["a"] * 4, max_null_rate=0.5)
    assert any(f.code == "OUTPUT_HIGH_NULL_RATE" for f in r.findings)
