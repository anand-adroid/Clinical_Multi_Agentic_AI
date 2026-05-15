"""Long-term memory endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.db.session import get_db
from backend.memory.long_term import LongTermMemory
from backend.schemas.api_schemas import MemoryPatternOut

router = APIRouter(prefix="/memory", tags=["memory"])


@router.get("/patterns", response_model=list[MemoryPatternOut])
def list_patterns(db: Session = Depends(get_db)) -> list[MemoryPatternOut]:
    ltm = LongTermMemory(db)
    return [
        MemoryPatternOut(
            id=p.id, target=p.target, signature=p.signature,
            rule_text=p.rule_text, code=p.code, sources=p.sources,
            score=p.score, times_used=p.times_used,
        )
        for p in ltm.all(limit=200)
    ]
