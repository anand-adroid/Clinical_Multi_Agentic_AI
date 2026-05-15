"""CSV-format spec adapter.

Real-world pharma derivation specs are almost never YAML — they live in
Excel or CSV files authored by clinical data managers. The parser here
accepts a permissive CSV format and converts it into the same internal
dict shape that ``backend.agents.spec_reviewer`` would receive from a
YAML upload, so downstream agents do not need to know which format the
author used.

Accepted columns (case-insensitive, several aliases each):

  name          REQUIRED   the derived column name
                aliases: column_name, derived_column, target

  sources       REQUIRED   raw column names (or other derived names)
                aliases: source, inputs
                separator: comma or pipe; quote the field if it contains commas

  rule          REQUIRED   plain-English transformation logic
                aliases: transformation, logic, description

  type          optional   default 'string'
                values: int | float | string | category | bool | date

  allowed_values  optional pipe- or comma-separated closed set for category type
                aliases: values, categories

  max_null_rate optional   default 0.5
                aliases: null_rate

  risk_class    optional   default 'routine'
                values: routine | critical | regulatory_critical | exploratory
                aliases: risk

  test_cases    optional   JSON array of {input: {...}, expected: ...} objects
                aliases: tests

Unknown columns are ignored. Empty rows are skipped. A row missing the
required ``name`` or ``rule`` fields raises ``CSVSpecError`` with the
offending row index so the author can correct it.
"""
from __future__ import annotations

import csv
import json
from io import StringIO
from typing import Any


class CSVSpecError(ValueError):
    """Raised when a CSV spec cannot be parsed into a valid internal spec."""


# Column-name aliases — keep them lowercased here and normalise inputs.
_NAME_KEYS = {"name", "column_name", "derived_column", "target"}
_SOURCES_KEYS = {"sources", "source", "inputs"}
_TYPE_KEYS = {"type"}
_ALLOWED_KEYS = {"allowed_values", "values", "categories"}
_RULE_KEYS = {"rule", "transformation", "logic", "description"}
_NULL_KEYS = {"max_null_rate", "null_rate"}
_RISK_KEYS = {"risk_class", "risk"}
_TESTS_KEYS = {"test_cases", "tests"}


def _normalise_key(k: str) -> str:
    return (k or "").strip().lower().replace(" ", "_")


def _find(row: dict, keys: set[str]) -> str | None:
    for k, v in row.items():
        if _normalise_key(str(k)) in keys:
            return v if v is None else str(v)
    return None


def _split_list(value: str | None) -> list[str]:
    if not value:
        return []
    raw = value.strip()
    if not raw:
        return []
    # Pipe takes precedence so authors can keep commas inside category labels.
    if "|" in raw:
        return [v.strip() for v in raw.split("|") if v.strip()]
    return [v.strip() for v in raw.split(",") if v.strip()]


def _parse_tests(value: str | None) -> list[dict[str, Any]]:
    """Test cases are JSON in a single cell because CSV cannot represent
    nested structures otherwise. Empty cells are treated as 'no test cases'.
    Malformed JSON is logged as an issue but does not fail the whole parse —
    the rest of the derivation is still useful."""
    if not value or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [t for t in parsed if isinstance(t, dict) and "input" in t and "expected" in t]


def _detect_delimiter(text: str) -> str:
    """CSV authored in Excel often uses ';' (locale-dependent). Try the
    sniffer; fall back to comma."""
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        return ","


def parse_csv_spec(
    csv_text: str,
    *,
    default_name: str = "Imported CSV specification",
) -> dict[str, Any]:
    """Parse a CSV string into the internal spec shape.

    Returns a dict with the same shape that YAML uploads produce::

        {
            "name": "...",
            "derivations": [{"name": ..., "sources": [...], ...}, ...],
        }

    Raises ``CSVSpecError`` on missing required columns / malformed rows.
    The error message is explicit about what was expected and includes a
    diagnostic when the file looks like a dataset rather than a spec —
    the most common authoring mistake by far.
    """
    if not csv_text or not csv_text.strip():
        raise CSVSpecError("CSV file is empty.")

    # Strip a UTF-8 BOM that Excel sometimes adds; the csv module does not
    # do this automatically and the BOM ends up prefixed onto the first
    # header name.
    if csv_text.startswith("﻿"):
        csv_text = csv_text.lstrip("﻿")

    delimiter = _detect_delimiter(csv_text)
    reader = csv.DictReader(StringIO(csv_text), delimiter=delimiter)
    if not reader.fieldnames:
        raise CSVSpecError("CSV file has no header row.")

    normalised_headers = {_normalise_key(h) for h in reader.fieldnames if h}

    # Single, opinionated diagnostic when the headers look nothing like a
    # spec. Most often this means the user swapped the dataset and spec
    # files in the upload form.
    dataset_smell = (
        "subject_id", "patient_id", "id",
        "age", "sex", "treatment_start_date", "visit_date",
        "ae_term", "ae_start_date", "drug_start_date", "max_intensity_score",
    )
    if not (normalised_headers & _NAME_KEYS) and any(
        h in normalised_headers for h in dataset_smell
    ):
        raise CSVSpecError(
            "This file looks like a dataset, not a derivation spec. "
            f"Detected columns: {sorted(normalised_headers)[:8]}. "
            "A spec CSV needs headers `name, sources, type, rule` (plus "
            "optional allowed_values, risk_class, max_null_rate). Confirm "
            "the dataset and spec uploads are not swapped."
        )

    if not (normalised_headers & _NAME_KEYS):
        raise CSVSpecError(
            "CSV is missing a required `name` column (accepted aliases: "
            "column_name, derived_column, target). "
            f"Headers detected: {sorted(normalised_headers)[:8]}."
        )
    if not (normalised_headers & _RULE_KEYS):
        raise CSVSpecError(
            "CSV is missing a required `rule` column (accepted aliases: "
            "transformation, logic, description). "
            f"Headers detected: {sorted(normalised_headers)[:8]}."
        )

    derivations: list[dict[str, Any]] = []
    issues: list[str] = []

    for row_index, row in enumerate(reader, start=2):  # row 2 = first data row
        name = (_find(row, _NAME_KEYS) or "").strip()
        if not name:
            continue  # quietly skip wholly empty rows
        rule = (_find(row, _RULE_KEYS) or "").strip()
        if not rule:
            issues.append(f"Row {row_index} ('{name}') has no rule; skipped.")
            continue
        sources = _split_list(_find(row, _SOURCES_KEYS))
        if not sources:
            # Not fatal — the Spec Reviewer will raise a clarification if it
            # cannot infer the sources from the rule text.
            issues.append(
                f"Row {row_index} ('{name}') has no declared sources; the "
                "Spec Reviewer will surface this as a clarification."
            )
        type_val = (_find(row, _TYPE_KEYS) or "string").strip().lower()
        allowed = _split_list(_find(row, _ALLOWED_KEYS))
        null_raw = _find(row, _NULL_KEYS)
        try:
            null_rate = float(null_raw) if null_raw and null_raw.strip() else 0.5
        except (TypeError, ValueError):
            null_rate = 0.5
            issues.append(
                f"Row {row_index} ('{name}'): max_null_rate '{null_raw}' is "
                "not numeric; defaulting to 0.5."
            )
        risk_class = (_find(row, _RISK_KEYS) or "routine").strip().lower()
        if risk_class not in (
            "exploratory", "routine", "critical", "regulatory_critical"
        ):
            issues.append(
                f"Row {row_index} ('{name}'): unknown risk_class "
                f"'{risk_class}'; defaulting to 'routine'."
            )
            risk_class = "routine"

        deriv: dict[str, Any] = {
            "name": name,
            "sources": sources,
            "type": type_val,
            "allowed_values": allowed,
            "rule": rule,
            "max_null_rate": null_rate,
            "risk_class": risk_class,
        }
        tests = _parse_tests(_find(row, _TESTS_KEYS))
        if tests:
            deriv["test_cases"] = tests
        derivations.append(deriv)

    if not derivations:
        raise CSVSpecError(
            "CSV file contained no usable derivations. "
            + ("Issues: " + "; ".join(issues) if issues else "")
        )

    spec: dict[str, Any] = {
        "name": default_name,
        "derivations": derivations,
    }
    if issues:
        spec["_csv_parse_notes"] = issues
    return spec
