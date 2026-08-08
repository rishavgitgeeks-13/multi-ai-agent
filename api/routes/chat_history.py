"""
Per-user chat history API (Mongo-backed via chat_history_service).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["ChatHistory"])


class ChatTurn(BaseModel):
    role: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None
    username: Optional[str] = None
    session_id: Optional[str] = None


class UserHistoryResponse(BaseModel):
    ok: bool = True
    username: str
    turns: List[ChatTurn] = Field(default_factory=list)
    total: int = 0


class ActivityResponse(BaseModel):
    ok: bool = True
    turns: List[ChatTurn] = Field(default_factory=list)
    total: int = 0


class AddTurnRequest(BaseModel):
    username: str
    role: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    session_id: str = ""


class AddTurnResponse(BaseModel):
    ok: bool
    message: str = ""


@router.get(
    "/users/{username}/history",
    response_model=UserHistoryResponse,
    summary="Get persisted chat history for a username",
)
async def get_user_chat_history(
    username: str,
    limit: int = Query(100, ge=1, le=500),
) -> UserHistoryResponse:
    user = (username or "").strip()
    if len(user) < 2:
        raise HTTPException(status_code=400, detail="Invalid username")
    try:
        from services.chat_history_service import get_user_history

        raw = get_user_history(user, limit=limit) or []
        turns = [
            ChatTurn(
                role=str(t.get("role") or ""),
                content=str(t.get("content") or ""),
                metadata=t.get("metadata") or {},
                created_at=str(t.get("created_at") or "") or None,
                username=str(t.get("username") or user),
                session_id=str(t.get("session_id") or ""),
            )
            for t in raw
        ]
        return UserHistoryResponse(
            ok=True, username=user.lower(), turns=turns, total=len(turns)
        )
    except Exception as exc:
        logger.warning("get_user_chat_history failed: %s", exc)
        return UserHistoryResponse(ok=True, username=user.lower(), turns=[], total=0)


@router.post(
    "/turns",
    response_model=AddTurnResponse,
    summary="Append a chat turn for a user",
)
async def add_chat_turn(body: AddTurnRequest) -> AddTurnResponse:
    try:
        from services.chat_history_service import add_turn

        ok = add_turn(
            username=body.username,
            role=body.role,
            content=body.content,
            metadata=body.metadata,
            session_id=body.session_id,
        )
        return AddTurnResponse(
            ok=ok,
            message="saved" if ok else "failed",
        )
    except Exception as exc:
        logger.warning("add_chat_turn failed: %s", exc)
        return AddTurnResponse(ok=False, message=str(exc))


@router.get(
    "/activity",
    response_model=ActivityResponse,
    summary="Recent activity across all users (admin / team visibility)",
)
async def get_team_activity(
    limit: int = Query(80, ge=1, le=500),
) -> ActivityResponse:
    try:
        from services.chat_history_service import get_all_recent_activity

        raw = get_all_recent_activity(limit=limit) or []
        turns = [
            ChatTurn(
                role=str(t.get("role") or ""),
                content=str(t.get("content") or ""),
                metadata=t.get("metadata") or {},
                created_at=str(t.get("created_at") or "") or None,
                username=str(t.get("username") or ""),
                session_id=str(t.get("session_id") or ""),
            )
            for t in raw
        ]
        return ActivityResponse(ok=True, turns=turns, total=len(turns))
    except Exception as exc:
        logger.warning("get_team_activity failed: %s", exc)
        return ActivityResponse(ok=True, turns=[], total=0)
