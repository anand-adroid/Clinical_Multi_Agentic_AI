"""DAG construction & topological ordering tests."""
from __future__ import annotations

import pytest

from backend.agents.dag_builder import DAGBuildError, _build_topo


def test_linear_chain_order():
    targets = {"A", "B", "C"}
    dag = {"A": ["src"], "B": ["A", "src"], "C": ["B"]}
    assert _build_topo(dag, targets) == ["A", "B", "C"]


def test_independent_targets_alphabetical():
    targets = {"Z", "A", "M"}
    dag = {"Z": ["src"], "A": ["src"], "M": ["src"]}
    assert _build_topo(dag, targets) == ["A", "M", "Z"]


def test_cycle_is_detected():
    targets = {"A", "B"}
    dag = {"A": ["B"], "B": ["A"]}
    with pytest.raises(DAGBuildError):
        _build_topo(dag, targets)
