"""
Long-Term Memory (LTM)
----------------------

What it stores
~~~~~~~~~~~~~~
Validated, reusable patterns. Each pattern is a tuple

    (signature, target, sources, rule_text, code, score, times_used)

`signature` is a stable hash of *the meaning of the rule*, not its exact
text. The idea: when a new derivation comes in whose rule paraphrases an old
one, the system should find the old code, not synthesise it again.

Why this matters
~~~~~~~~~~~~~~~~
* **Consistency** — across studies a phrase like "age >= 65 = elderly" should
  always produce the same derivation. LTM enforces that.
* **Cost** — re-using a vetted snippet is free; calling an LLM is not.
* **Learning from HITL** — every approved edit becomes a new pattern, so the
  reviewer's correction propagates into future runs.

Retrieval
~~~~~~~~~
The retrieval scheme is deliberately simple: the signature is the SHA-256
of canonical tokens (normalised target name + sorted source names +
tokenised rule text). A production deployment that needs semantic recall
of paraphrased rules can swap the signature for an embedding index — the
``LongTermMemory`` interface stays the same.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable

from sqlalchemy.orm import Session

from backend.db.repositories import ClarificationMemoryRepository, MemoryRepository
from backend.utils.logging_setup import get_logger

log = get_logger("memory.ltm")

_TOKEN = re.compile(r"[a-zA-Z0-9_]+")


def _normalise_tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text or "")]


def compute_signature(
    target: str,
    sources: Iterable[str],
    rule_text: str,
    *,
    generator_version: str = "",
) -> str:
    """Stable signature for a (target, sources, rule, generator-version) tuple.

    Including ``generator_version`` means a code-generator prompt bump
    automatically invalidates previously-cached patterns instead of silently
    serving stale code. The downside is a one-time LLM hit on the first run
    after a prompt upgrade; the upside is the cache cannot mask a prompt
    regression.
    """
    tokens = sorted(_normalise_tokens(rule_text))
    payload = "|".join([
        target.lower(),
        ",".join(sorted(s.lower() for s in sources)),
        " ".join(tokens),
        generator_version,
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def compute_clarification_signature(target: str, issue: str) -> str:
    """Phase 2: signature for the clarification-memory store. Keyed by target
    name + tokenised issue text, so paraphrased issues map to the same row."""
    payload = "|".join([
        target.lower(),
        " ".join(sorted(_normalise_tokens(issue))),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


@dataclass
class RetrievedClarification:
    id: int
    target: str
    issue: str
    answer: str
    score: float
    times_used: int


@dataclass
class RetrievedPattern:
    id: int
    signature: str
    target: str
    sources: list[str]
    rule_text: str
    code: str
    score: float
    reasoning: str | None = None


class LongTermMemory:
    def __init__(self, db: Session) -> None:
        self.repo = MemoryRepository(db)
        self.clarif_repo = ClarificationMemoryRepository(db)

    def lookup(
        self, *, target: str, sources: Iterable[str], rule_text: str,
        generator_version: str = "",
    ) -> RetrievedPattern | None:
        sig = compute_signature(
            target, sources, rule_text,
            generator_version=generator_version,
        )
        pat = self.repo.find_for(signature=sig, target=target)
        if pat:
            log.info("ltm.hit", target=target, signature=sig, score=pat.score)
            return RetrievedPattern(
                id=pat.id, signature=pat.signature, target=pat.target,
                sources=list(pat.sources), rule_text=pat.rule_text,
                code=pat.code, score=pat.score,
                reasoning=pat.reasoning,
            )
        log.info("ltm.miss", target=target, signature=sig)
        return None

    def remember(self, *, target: str, sources: Iterable[str], rule_text: str,
                 code: str, created_by: str = "system",
                 reasoning: str | None = None,
                 generator_version: str = "") -> None:
        sig = compute_signature(
            target, sources, rule_text,
            generator_version=generator_version,
        )
        self.repo.add_pattern(
            signature=sig, target=target, rule_text=rule_text,
            code=code, sources=list(sources), created_by=created_by,
            reasoning=reasoning,
        )
        log.info("ltm.write", target=target, signature=sig, source=created_by)

    def reinforce(self, pattern_id: int) -> None:
        self.repo.mark_used(pattern_id)

    def all(self, limit: int = 100):
        return self.repo.list(limit=limit)

    # ------------------------------------------------------------- Phase 2
    def lookup_clarification(
        self, *, target: str, issue: str
    ) -> RetrievedClarification | None:
        sig = compute_clarification_signature(target, issue)
        row = self.clarif_repo.find_for(signature=sig, target=target)
        if not row:
            log.info("clarif.miss", target=target, signature=sig)
            return None
        log.info("clarif.hit", target=target, signature=sig, score=row.score)
        return RetrievedClarification(
            id=row.id, target=row.target, issue=row.issue,
            answer=row.answer, score=row.score, times_used=row.times_used,
        )

    def remember_clarification(
        self, *, target: str, issue: str, answer: str, reviewer: str = "system"
    ) -> None:
        sig = compute_clarification_signature(target, issue)
        self.clarif_repo.upsert(
            signature=sig, target=target, issue=issue,
            answer=answer, reviewer=reviewer,
        )
        log.info("clarif.write", target=target, signature=sig, reviewer=reviewer)

    def reinforce_clarification(self, clarif_id: int) -> None:
        self.clarif_repo.mark_used(clarif_id)
