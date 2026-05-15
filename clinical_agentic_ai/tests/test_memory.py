"""Long-term memory tests."""
from __future__ import annotations

from backend.db.session import session_scope
from backend.memory.long_term import LongTermMemory, compute_signature


def test_signature_is_stable():
    s1 = compute_signature("AGE_GROUP", ["age"], "Bucket age into <18, 18-64, >=65.")
    s2 = compute_signature("AGE_GROUP", ["age"], "bucket age into <18,  18-64, >=65")
    # Word-set is identical → same signature.
    assert s1 == s2


def test_remember_and_lookup_roundtrip():
    with session_scope() as db:
        ltm = LongTermMemory(db)
        ltm.remember(
            target="AGE_GROUP", sources=["age"],
            rule_text="Bucket age.",
            code="def derive(row): return None",
            created_by="test",
        )
    with session_scope() as db:
        ltm = LongTermMemory(db)
        hit = ltm.lookup(target="AGE_GROUP", sources=["age"], rule_text="bucket age")
        assert hit is not None
        assert hit.target == "AGE_GROUP"
