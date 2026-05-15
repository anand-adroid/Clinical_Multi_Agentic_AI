"""
Deterministic content hashing — the backbone of reproducibility.

Every artifact (spec, dataset, generated code, derived column) is hashed
and the hash is recorded in the audit trail. Two runs on the same inputs
MUST produce identical hashes; that is the reproducibility proof a
regulator needs.
"""
from __future__ import annotations

import hashlib
import io
import json
from typing import Any

import pandas as pd


def _canonical(obj: Any) -> bytes:
    """Canonical JSON — stable ordering, no insignificant whitespace."""
    return json.dumps(obj, sort_keys=True, default=str, separators=(",", ":")).encode()


def hash_obj(obj: Any) -> str:
    return hashlib.sha256(_canonical(obj)).hexdigest()


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_dataframe(df: pd.DataFrame) -> str:
    """Stable across runs as long as columns, dtypes, and row order match."""
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)  # canonical binary form
    return hashlib.sha256(buf.getvalue()).hexdigest()


def short(h: str, n: int = 12) -> str:
    return h[:n]
