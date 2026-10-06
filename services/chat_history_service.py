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


def _migrate_file_to_mongo_if_empty(col) -> None:
    """One-time seed: copy local file history into Mongo when the collection is empty."""
    try:
        if col.estimated_document_count() > 0:
            return
        data = _load_file()
        if not data:
            return
        docs: List[Dict[str, Any]] = []
        for turns in data.values():
            for t in turns or []:
                if isinstance(t, dict) and t.get("content"):
                    docs.append(dict(t))
        if not docs:
            return
        col.insert_many(docs, ordered=False)
        logger.info("Migrated %d chat history turns from file → Mongo", len(docs))
    except Exception as exc:
        logger.warning("chat history file→Mongo migrate skipped: %s", exc)


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
    """Append one turn for a user. Writes Mongo + file (dual) for durability."""
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

    mongo_ok = False
    col = _mongo_collection()
    if col is not None:
        try:
            col.insert_one(dict(doc))
            mongo_ok = True
        except Exception as exc:
            logger.warning("Mongo add_turn failed, continuing with file: %s", exc)

    file_ok = False
    try:
        data = _load_file()
        bucket = data.setdefault(user, [])
        bucket.append(doc)
        # Cap per-user file history
        if len(bucket) > 500:
            data[user] = bucket[-500:]
        _save_file(data)
        file_ok = True
    except Exception as exc:
        logger.error("chat history file save failed: %s", exc)

    return mongo_ok or file_ok


def get_user_history(username: str, limit: int = 100) -> List[Dict[str, Any]]:
    """Return oldest-first turns for one user (merges email + local-part aliases)."""
    aliases = _username_aliases(username)
    if not aliases:
        return []
    cap = max(1, min(limit, 500))

    collected: List[Dict[str, Any]] = []
    seen_keys: set[str] = set()

    def _absorb(docs: List[Dict[str, Any]]) -> None:
        for t in docs or []:
            if not isinstance(t, dict) or not t.get("content"):
                continue
            key = (
                f"{t.get('created_at')}|{t.get('role')}|{str(t.get('content'))[:80]}"
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            collected.append(t)

    col = _mongo_collection()
    if col is not None:
        _migrate_file_to_mongo_if_empty(col)
        try:
            cursor = (
                col.find({"username": {"$in": aliases}}, {"_id": 0})
                .sort("created_at", 1)
                .limit(cap * 2)
            )
            _absorb(list(cursor))
        except Exception as exc:
            logger.warning("Mongo get_user_history failed: %s", exc)

    data = _load_file()
    for alias in aliases:
        _absorb(data.get(alias) or [])

    collected.sort(key=lambda t: str(t.get("created_at") or ""))
    return collected[-cap:]


def _username_aliases(username: str) -> List[str]:
    """Match both name@domain and local-part history keys."""
    raw = (username or "").strip().lower()
    if not raw:
        return []
    out = [raw]
    if "@" in raw:
        local = raw.split("@", 1)[0]
        if local and local not in out:
            out.append(local)
    else:
        # Common company domain for this app
        email = f"{raw}@gitgeeks.com"
        if email not in out:
            out.insert(0, email)
    return out


def get_all_recent_activity(limit: int = 100) -> List[Dict[str, Any]]:
    """Recent turns across all users (newest first) for admin activity view."""
    cap = max(1, min(limit, 500))
    col = _mongo_collection()
    if col is not None:
        try:
            cursor = (
                col.find({}, {"_id": 0})
                .sort("created_at", -1)
                .limit(cap)
            )
            docs = list(cursor)
            if docs:
                return docs
        except Exception as exc:
            logger.warning("Mongo get_all_recent_activity failed: %s", exc)

    data = _load_file()
    all_turns: List[Dict[str, Any]] = []
    for turns in data.values():
        all_turns.extend(turns)
    all_turns.sort(key=lambda t: str(t.get("created_at") or ""), reverse=True)
    return all_turns[:cap]


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


def list_history_usernames() -> List[str]:
    """All usernames that have at least one saved chat turn."""
    names: set[str] = set()
    col = _mongo_collection()
    if col is not None:
        try:
            for value in col.distinct("username"):
                key = str(value or "").strip().lower()
                if key:
                    names.add(key)
        except Exception as exc:
            logger.warning("Mongo list_history_usernames failed: %s", exc)

    for key in _load_file().keys():
        user = str(key or "").strip().lower()
        if user:
            names.add(user)
    return sorted(names)


def get_user_usage_summary(username: str) -> Dict[str, Any]:
    """
    Lightweight usage stats for admin monitoring.

    Returns turn_count, conversation_count, last_activity (ISO), username,
    plus last-7-day counts for weekly reporting.
    """
    user = (username or "").strip().lower()
    empty = {
        "username": user,
        "turn_count": 0,
        "conversation_count": 0,
        "last_activity": "",
        "week_turn_count": 0,
        "week_conversation_count": 0,
    }
    if not user:
        return empty

    turns = get_user_history(user, limit=500)
    if not turns:
        return empty

    last_at = ""
    for t in turns:
        created = str((t or {}).get("created_at") or "")
        if created > last_at:
            last_at = created

    conversations = turns_to_conversations(turns)
    week_turns = _filter_turns_since(turns, days=7)
    week_convs = turns_to_conversations(week_turns)
    return {
        "username": user,
        "turn_count": len(turns),
        "conversation_count": len(conversations),
        "last_activity": last_at,
        "week_turn_count": len(week_turns),
        "week_conversation_count": len(week_convs),
    }


def _filter_turns_since(turns: List[Dict[str, Any]], days: int = 7) -> List[Dict[str, Any]]:
    """Keep turns whose created_at is within the last N days (UTC)."""
    from datetime import timedelta

    if days <= 0:
        return list(turns or [])
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    cutoff_iso = cutoff.isoformat()
    out: List[Dict[str, Any]] = []
    for t in turns or []:
        created = str((t or {}).get("created_at") or "")
        if not created:
            continue
        # Accept both Z and +00:00 style timestamps
        norm = created.replace("Z", "+00:00")
        try:
            # Fast path: ISO string compare works when both are timezone-aware ISO
            if norm >= cutoff_iso[:19]:
                # Prefer parsed compare when possible
                try:
                    ts = datetime.fromisoformat(norm)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    if ts >= cutoff:
                        out.append(t)
                    continue
                except Exception:
                    out.append(t)
                    continue
        except Exception:
            continue
        # Fallback string compare for sortable ISO prefixes
        if created[:10] >= cutoff.date().isoformat():
            out.append(t)
    return out


def build_portal_usage_report(days: int = 7) -> List[Dict[str, Any]]:
    """
    Admin weekly usage report across all known users.

    Rows include signup roster + anyone with chat history.
    Safe for CSV export.
    """
    days = max(1, min(int(days or 7), 90))
    names: set[str] = set(list_history_usernames())
    try:
        from services.user_auth import list_users

        for row in list_users() or []:
            email = str(row.get("username") or row.get("email") or "").strip().lower()
            if email:
                names.add(email)
    except Exception as exc:
        logger.warning("usage report list_users failed: %s", exc)

    rows: List[Dict[str, Any]] = []
    for name in sorted(names):
        turns = get_user_history(name, limit=500)
        conversations = turns_to_conversations(turns)
        period_turns = _filter_turns_since(turns, days=days)
        period_convs = turns_to_conversations(period_turns)
        last_at = ""
        for t in turns:
            created = str((t or {}).get("created_at") or "")
            if created > last_at:
                last_at = created
        status = ""
        try:
            from services.user_auth import _get_user, _user_status

            user = _get_user(name)
            if user:
                status = _user_status(user)
        except Exception:
            status = ""
        rows.append(
            {
                "username": name,
                "status": status or "active",
                "conversations_all_time": len(conversations),
                "turns_all_time": len(turns),
                f"conversations_last_{days}d": len(period_convs),
                f"turns_last_{days}d": len(period_turns),
                "last_activity": last_at,
            }
        )
    turn_key = f"turns_last_{days}d"
    rows.sort(
        key=lambda r: (int(r.get(turn_key) or 0), str(r.get("last_activity") or "")),
        reverse=True,
    )
    return rows


def usage_report_csv(days: int = 7) -> str:
    """CSV text for weekly portal usage report."""
    import csv
    import io

    rows = build_portal_usage_report(days=days)
    if not rows:
        return "username,status,conversations_all_time,turns_all_time,last_activity\n"
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()
