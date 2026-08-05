"""
Session / chat-history API routes.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sessions", tags=["Sessions"])


class HistoryTurn(BaseModel):
    role: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None


class HistoryResponse(BaseModel):
    ok: bool = True
    session_id: str
    turns: List[HistoryTurn] = Field(default_factory=list)
    total: int = 0


@router.get(
    "/{session_id}/history",
    response_model=HistoryResponse,
    summary="Get conversation history for a session",
)
async def get_session_history(
    session_id: str,
    limit: int = Query(50, ge=1, le=200),
) -> HistoryResponse:
    """Return stored chat/workflow turns for the session (oldest first)."""
    if not session_id or len(session_id) < 4:
        raise HTTPException(status_code=400, detail="Invalid session_id")

    try:
        from memory.conversation_memory import ConversationMemory

        mem = ConversationMemory(session_id=session_id, max_history=limit)
        raw = mem.get_history(limit=limit) or []
        turns: List[HistoryTurn] = []
        for turn in raw:
            created = turn.get("created_at")
            if hasattr(created, "isoformat"):
                created = created.isoformat()
            turns.append(
                HistoryTurn(
                    role=str(turn.get("role") or ""),
                    content=str(turn.get("content") or ""),
                    metadata=turn.get("metadata") or {},
                    created_at=str(created) if created else None,
                )
            )
        return HistoryResponse(
            ok=True,
            session_id=session_id,
            turns=turns,
            total=len(turns),
        )
    except Exception as exc:
        logger.warning("Session history load failed: %s", exc)
        # Soft fail — UI can still use local session history
        return HistoryResponse(
            ok=True,
            session_id=session_id,
            turns=[],
            total=0,
        )
