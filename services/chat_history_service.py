"""
Per-user chat history persistence (MongoDB + JSON file fallback).

History is keyed by username so the same login always sees prior turns.
Admin can list recent activity across users.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_FILE_PATH = Path(__file__).resolve().parent.parent / "data" / "chat_history.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mongo_collection():
    try:
        from memory.mongodb import MongoDBClient
        from pymongo import ASCENDING, DESCENDING

        client = MongoDBClient()
        col = client.db.user_chat_turns
        col.create_index([("username", ASCENDING), ("created_at", ASCENDING)])
        col.create_index([("created_at", DESCENDING)])
        return col
    except Exception as exc:
        logger.info("Mongo chat history unavailable, using file fallback: %s", exc)
        return None


def _load_file() -> Dict[str, List[Dict[str, Any]]]:
    if not _FILE_PATH.exists():
        return {}
    try:
        data = json.loads(_FILE_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception as exc:
        logger.warning("chat history file read failed: %s", exc)
    return {}


def _save_file(data: Dict[str, List[Dict[str, Any]]]) -> None:
    _FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _FILE_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def add_turn(
    username: str,
    role: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    session_id: str = "",
) -> bool:
    """Append one turn for a user. Returns True on success."""
    user = (username or "").strip().lower()
    if not user or not content:
        return False
    doc = {
        "username": user,
        "session_id": session_id or f"user-{user}",
        "role": role,
        "content": content,
        "metadata": metadata or {},
        "created_at": _now_iso(),
    }
    col = _mongo_collection()
    if col is not None:
        try:
            col.insert_one(dict(doc))
            return True
        except Exception as exc:
            logger.warning("Mongo add_turn failed, file fallback: %s", exc)

    data = _load_file()
    bucket = data.setdefault(user, [])
    bucket.append(doc)
    # Cap per-user file history
    if len(bucket) > 500:
        data[user] = bucket[-500:]
    try:
        _save_file(data)
        return True
    except Exception as exc:
        logger.error("chat history file save failed: %s", exc)
        return False


def get_user_history(username: str, limit: int = 100) -> List[Dict[str, Any]]:
    """Return oldest-first turns for one user."""
    user = (username or "").strip().lower()
    if not user:
        return []
    col = _mongo_collection()
    if col is not None:
        try:
            cursor = (
                col.find({"username": user}, {"_id": 0})
                .sort("created_at", 1)
                .limit(max(1, min(limit, 500)))
            )
            return list(cursor)
        except Exception as exc:
            logger.warning("Mongo get_user_history failed: %s", exc)

    bucket = _load_file().get(user) or []
    return bucket[-max(1, min(limit, 500)) :]


def get_all_recent_activity(limit: int = 100) -> List[Dict[str, Any]]:
    """Recent turns across all users (newest first) for admin activity view."""
    col = _mongo_collection()
    if col is not None:
        try:
            cursor = (
                col.find({}, {"_id": 0})
                .sort("created_at", -1)
                .limit(max(1, min(limit, 500)))
            )
            return list(cursor)
        except Exception as exc:
            logger.warning("Mongo get_all_recent_activity failed: %s", exc)

    data = _load_file()
    all_turns: List[Dict[str, Any]] = []
    for turns in data.values():
        all_turns.extend(turns)
    all_turns.sort(key=lambda t: str(t.get("created_at") or ""), reverse=True)
    return all_turns[: max(1, min(limit, 500))]


def stable_session_id(username: str) -> str:
    """Deterministic session id so ConversationMemory also persists per user."""
    user = (username or "").strip().lower() or "anonymous"
    return f"user-{user}"


def make_conversation_title(prompt: str, max_len: int = 72) -> str:
    """Short ChatGPT-style title from the user prompt."""
    text = " ".join((prompt or "").strip().split())
    if not text:
        return "Untitled generation"
    if len(text) <= max_len:
        return text
    cut = text[: max_len - 1].rsplit(" ", 1)[0] or text[: max_len - 1]
    return cut + "…"


def turns_to_conversations(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Collapse raw turns into conversation cards (newest-first).

    Supports:
      - role=conversation (new format: one card per generation)
      - legacy user + assistant pairs
    """
    if not turns:
        return []

    # Pair in chronological order (team activity may arrive newest-first)
    ordered = sorted(
        list(turns),
        key=lambda t: str((t or {}).get("created_at") or ""),
    )

    conversations: List[Dict[str, Any]] = []
    i = 0
    n = len(ordered)
    while i < n:
        t = ordered[i] or {}
        role = str(t.get("role") or "").lower()
        meta = t.get("metadata") or {}
        created = str(t.get("created_at") or "")
        username = str(t.get("username") or "")

        if role == "conversation":
            prompt = str(meta.get("prompt") or t.get("content") or "")
            conversations.append(
                {
                    "id": str(meta.get("conversation_id") or f"conv-{i}-{created}"),
                    "title": str(meta.get("title") or make_conversation_title(prompt)),
                    "prompt": prompt,
                    "workflow": str(meta.get("workflow") or "auto"),
                    "score": meta.get("score"),
                    "status": meta.get("status"),
                    "ok": meta.get("ok", True),
                    "result": meta.get("result"),
                    "preview": str(t.get("content") or "")[:400],
                    "created_at": created,
                    "username": username,
                    "hashtags": meta.get("hashtags") or [],
                }
            )
            i += 1
            continue

        if role == "user":
            prompt = str(t.get("content") or "")
            assistant = None
            if (
                i + 1 < n
                and str((ordered[i + 1] or {}).get("role") or "").lower()
                == "assistant"
            ):
                assistant = ordered[i + 1]
                i += 2
            else:
                i += 1
            a_meta = (assistant or {}).get("metadata") or {}
            a_content = str((assistant or {}).get("content") or "")
            conversations.append(
                {
                    "id": f"legacy-{created}-{i}",
                    "title": make_conversation_title(prompt),
                    "prompt": prompt,
                    "workflow": str(
                        a_meta.get("workflow")
                        or meta.get("workflow")
                        or "auto"
                    ),
                    "score": a_meta.get("score"),
                    "status": a_meta.get("status"),
                    "ok": a_meta.get("ok", bool(assistant)),
                    "result": a_meta.get("result"),
                    "preview": a_content[:400],
                    "assistant_text": a_content,
                    "created_at": created
                    or str((assistant or {}).get("created_at") or ""),
                    "username": username
                    or str((assistant or {}).get("username") or ""),
                    "hashtags": a_meta.get("hashtags") or [],
                }
            )
            continue

        # Orphan assistant / other roles — skip as standalone cards
        i += 1

    conversations.sort(key=lambda c: str(c.get("created_at") or ""), reverse=True)
    return conversations


def group_conversations_by_user(
    conversations: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Group conversation cards by username (display order: alpha, then recency)."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for conv in conversations or []:
        who = str((conv or {}).get("username") or "unknown").strip().lower() or "unknown"
        groups.setdefault(who, []).append(conv)
    # Keep each user's list newest-first (already sorted globally)
    for who in groups:
        groups[who].sort(key=lambda c: str(c.get("created_at") or ""), reverse=True)
    return dict(sorted(groups.items(), key=lambda kv: kv[0]))
