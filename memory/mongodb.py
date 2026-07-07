"""
MongoDB client and repositories for persisting conversation history,
workflow run snapshots, and session metadata.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from config.settings import settings

logger = logging.getLogger(__name__)


# ==========================================================================
# Singleton MongoDB connection
# ==========================================================================


class MongoDBClient:
    """Thread-safe singleton that holds the MongoClient and database handle."""

    _instance: "MongoDBClient" = None
    _client: MongoClient = None
    _db: Database = None

    def __new__(cls) -> "MongoDBClient":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._connect()
        return cls._instance

    def _connect(self) -> None:
        self._client = MongoClient(settings.MONGODB_URI)
        self._db = self._client[settings.MONGODB_DATABASE]
        self._ensure_indexes()
        logger.info("MongoDB connected — database: %s", settings.MONGODB_DATABASE)

    @property
    def db(self) -> Database:
        return self._db

    def _ensure_indexes(self) -> None:
        # ---- conversations ------------------------------------------------
        convs: Collection = self._db.conversations
        convs.create_index([("session_id", ASCENDING)])
        convs.create_index([("created_at", DESCENDING)])
        convs.create_index([("session_id", ASCENDING), ("created_at", ASCENDING)])

        # ---- workflow_runs ------------------------------------------------
        runs: Collection = self._db.workflow_runs
        runs.create_index([("request_id", ASCENDING)], unique=True, sparse=True)
        runs.create_index([("session_id", ASCENDING)])
        runs.create_index([("brand", ASCENDING)])
        runs.create_index([("created_at", DESCENDING)])
        runs.create_index([("workflow_status", ASCENDING)])

        # ---- sessions -----------------------------------------------------
        sessions: Collection = self._db.sessions
        sessions.create_index([("session_id", ASCENDING)], unique=True)
        sessions.create_index([("last_active", DESCENDING)])



# ==========================================================================
# Repository: Conversations
# ==========================================================================


class ConversationRepository:
    """CRUD operations for per-session conversation turns."""

    def __init__(self, client: Optional[MongoDBClient] = None) -> None:
        self._collection: Collection = (client or MongoDBClient()).db.conversations

    def save_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Persist one conversation turn. Returns the inserted document id."""
        doc = {
            "session_id": session_id,
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "created_at": datetime.now(timezone.utc),
        }
        result = self._collection.insert_one(doc)
        return str(result.inserted_id)

    def get_history(
        self,
        session_id: str,
        limit: int = 20,
        roles: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Return up to `limit` turns for the session, oldest first."""
        query: Dict[str, Any] = {"session_id": session_id}
        if roles:
            query["role"] = {"$in": roles}

        cursor = (
            self._collection
            .find(query, {"_id": 0})
            .sort("created_at", ASCENDING)
            .limit(limit)
        )
        return list(cursor)

    def delete_session(self, session_id: str) -> int:
        """Delete all turns for a session. Returns the number of deleted docs."""
        result = self._collection.delete_many({"session_id": session_id})
        return result.deleted_count

    def count_turns(self, session_id: str) -> int:
        return self._collection.count_documents({"session_id": session_id})


# ==========================================================================
# Repository: Workflow Runs
# ==========================================================================


class WorkflowRunRepository:
    """Persist and query full ContentState snapshots from each workflow run."""

    def __init__(self, client: Optional[MongoDBClient] = None) -> None:
        self._collection: Collection = (client or MongoDBClient()).db.workflow_runs

    def save_run(self, state: Dict[str, Any]) -> str:
        """
        Upsert a workflow run keyed on request_id.
        Returns the request_id (or bson id for brand-new inserts).
        """
        request_id = state.get("request_id")
        now = datetime.now(timezone.utc)
        doc = {
            "request_id": request_id,
            "session_id": state.get("session_id"),
            "brand": state.get("brand_context", {}).get("brand"),
            "content_type": state.get("content_type"),
            "platform": state.get("platform"),
            "objective": state.get("objective"),
            "user_input": state.get("user_input"),
            "workflow_status": state.get("workflow_status"),
            "review_score": state.get("review", {}).get("score"),
            "revision_count": state.get("revision_count", 0),
            "final_output": state.get("final_output"),
            "errors": state.get("errors", []),
            "updated_at": now,
            "state_snapshot": state,
        }
        result = self._collection.update_one(
            {"request_id": request_id},
            {
                "$set": doc,
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
        )
        return str(result.upserted_id) if result.upserted_id else str(request_id)

    def get_run(self, request_id: str) -> Optional[Dict[str, Any]]:
        return self._collection.find_one({"request_id": request_id}, {"_id": 0})

    def list_runs(
        self,
        brand: Optional[str] = None,
        content_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """List recent runs with optional filters. Excludes the raw state snapshot."""
        query: Dict[str, Any] = {}
        if brand:
            query["brand"] = brand
        if content_type:
            query["content_type"] = content_type
        if status:
            query["workflow_status"] = status

        cursor = (
            self._collection
            .find(query, {"_id": 0, "state_snapshot": 0})
            .sort("updated_at", DESCENDING)
            .limit(limit)
        )
        return list(cursor)


# ==========================================================================
# Repository: Sessions
# ==========================================================================


class SessionRepository:
    """Track user sessions and their metadata."""

    def __init__(self, client: Optional[MongoDBClient] = None) -> None:
        self._collection: Collection = (client or MongoDBClient()).db.sessions

    def upsert_session(
        self,
        session_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create or refresh a session. Returns the current session document."""
        now = datetime.now(timezone.utc)
        self._collection.update_one(
            {"session_id": session_id},
            {
                "$set": {"last_active": now, **(metadata or {})},
                "$setOnInsert": {"session_id": session_id, "created_at": now},
            },
            upsert=True,
        )
        return self.get_session(session_id)

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self._collection.find_one({"session_id": session_id}, {"_id": 0})

    def list_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        cursor = (
            self._collection
            .find({}, {"_id": 0})
            .sort("last_active", DESCENDING)
            .limit(limit)
        )
        return list(cursor)
