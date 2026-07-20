"""
High-level conversation memory interface for LangGraph agents.

Storage split:
  - MongoDB  → conversation turns, workflow-run snapshots, session metadata
  - Pinecone → semantic vector embeddings for similarity search
"""

import logging
import uuid
from typing import Any, Dict, List, Optional

from memory.mongodb import (
    ConversationRepository,
    MongoDBClient,
    SessionRepository,
    WorkflowRunRepository,
)
from memory.vector import VectorStore

logger = logging.getLogger(__name__)

_CONV_NS_PREFIX = "conv"


class ConversationMemory:
    """
    Central memory interface consumed by LangGraph agents.

    Example usage
    -------------
    memory = ConversationMemory(session_id="abc-123")

    # Store turns
    memory.add_user_message("Write an article about AI.")
    memory.add_assistant_message("Here is your article...")

    # Retrieve history for the LLM prompt
    messages = memory.get_formatted_history(limit=10)

    # Find semantically similar past messages
    similar = memory.search_similar("machine learning automation")

    # Persist the full LangGraph state after a run
    memory.save_workflow_state(state)
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        max_history: int = 20,
    ) -> None:
        self.session_id: str = session_id or str(uuid.uuid4())
        self.max_history: int = max_history

        client = MongoDBClient()
        self._convs = ConversationRepository(client)
        self._runs = WorkflowRunRepository(client)
        self._sessions = SessionRepository(client)
        self._vectors = VectorStore()

        # Touch the session so it exists in MongoDB
        self._sessions.upsert_session(self.session_id)

        logger.info("ConversationMemory initialised — session_id=%s", self.session_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _vector_namespace(self) -> str:
        """Namespace that scopes vector embeddings to this session."""
        return f"{_CONV_NS_PREFIX}:{self.session_id}"

    def _index_message(
        self,
        content: str,
        role: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Silently embed and store a message for later semantic search."""
        try:
            self._vectors.add_document(
                text=content,
                namespace=self._vector_namespace,
                doc_type="conversation",
                metadata={"role": role, **(metadata or {})},
            )
        except Exception as exc:
            logger.warning("Vector indexing failed for session %s: %s", self.session_id, exc)

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def update_session_metadata(self, metadata: Dict[str, Any]) -> None:
        """Attach arbitrary metadata (brand, user_id, etc.) to the session."""
        self._sessions.upsert_session(self.session_id, metadata)

    def get_session(self) -> Optional[Dict[str, Any]]:
        return self._sessions.get_session(self.session_id)

    # ------------------------------------------------------------------
    # Storing conversation turns
    # ------------------------------------------------------------------

    def add_user_message(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Persist a user turn and index it for semantic search."""
        turn_id = self._convs.save_turn(
            session_id=self.session_id,
            role="user",
            content=content,
            metadata=metadata,
        )
        self._index_message(content, role="user", metadata=metadata)
        return turn_id

    def add_assistant_message(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Persist an assistant turn and index it for semantic search."""
        turn_id = self._convs.save_turn(
            session_id=self.session_id,
            role="assistant",
            content=content,
            metadata=metadata,
        )
        self._index_message(content, role="assistant", metadata=metadata)
        return turn_id

    def add_system_message(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Persist a system turn (not indexed for semantic search)."""
        return self._convs.save_turn(
            session_id=self.session_id,
            role="system",
            content=content,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Retrieving conversation history
    # ------------------------------------------------------------------

    def get_history(
        self,
        limit: Optional[int] = None,
        roles: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return raw turn documents (with role, content, metadata, created_at)
        ordered oldest-first.
        """
        return self._convs.get_history(
            session_id=self.session_id,
            limit=limit or self.max_history,
            roles=roles,
        )

    def get_formatted_history(
        self,
        limit: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """
        Return history as [{"role": ..., "content": ...}] — ready for
        direct use as the `messages` argument in OpenAI / LangChain calls.
        """
        return [
            {"role": turn["role"], "content": turn["content"]}
            for turn in self.get_history(limit=limit)
        ]

    def clear(self) -> int:
        """
        Delete all turns and vector embeddings for this session.
        Returns the number of deleted turn documents.
        """
        deleted = self._convs.delete_session(self.session_id)
        self._vectors.delete_namespace(self._vector_namespace)
        logger.info("Cleared session %s — %d turns removed", self.session_id, deleted)
        return deleted

    # ------------------------------------------------------------------
    # Semantic search over past messages
    # ------------------------------------------------------------------

    def search_similar(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Find the `top_k` past messages in this session that are most
        semantically similar to `query`.

        Returns list of {text, metadata, score} dicts.
        """
        return self._vectors.similarity_search(
            query=query,
            namespace=self._vector_namespace,
            top_k=top_k,
            score_threshold=score_threshold,
        )

    # ------------------------------------------------------------------
    # Workflow state persistence
    # ------------------------------------------------------------------

    def save_workflow_state(self, state: Dict[str, Any]) -> str:
        """
        Persist a ContentState snapshot to MongoDB.
        Automatically attaches the current session_id.
        Returns the request_id (or bson id for new inserts).
        """
        enriched = {**state, "session_id": self.session_id}
        run_id = self._runs.save_run(enriched)
        logger.info(
            "Saved workflow run request_id=%s session_id=%s",
            state.get("request_id"),
            self.session_id,
        )
        return run_id

    def get_workflow_run(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single workflow run snapshot by its request_id."""
        return self._runs.get_run(request_id)

    def get_past_runs(
        self,
        brand: Optional[str] = None,
        content_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        List recent workflow runs (without the raw state snapshot).
        Useful for surfacing past outputs to agents as context.
        """
        return self._runs.list_runs(
            brand=brand,
            content_type=content_type,
            status=status,
            limit=limit,
        )
