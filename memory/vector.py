"""
Vector store backed by Pinecone for semantic search over knowledge-base
documents and conversation embeddings.

Embeddings are produced by models.embeddings.EmbeddingProvider
(OpenAI text-embedding-3-small, 1536-dimensional, L2-normalized).

Pinecone namespaces map 1-to-1 with the `namespace` argument so that
brand knowledge bases and per-session conversation vectors stay isolated.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional

from pinecone import Pinecone

from config.settings import settings
from models.embeddings import embedding_provider

logger = logging.getLogger(__name__)


# ==========================================================================
# Pinecone singleton
# ==========================================================================


class _PineconeClient:
    """Lazy singleton that holds the Pinecone Index handle."""

    _instance: "_PineconeClient" = None
    _index = None

    def __new__(cls) -> "_PineconeClient":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._connect()
        return cls._instance

    def _connect(self) -> None:
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self._index = pc.Index(settings.PINECONE_INDEX_NAME)
        logger.info("Pinecone connected — index: %s", settings.PINECONE_INDEX_NAME)

    @property
    def index(self):
        return self._index


# ==========================================================================
# VectorStore
# ==========================================================================


class VectorStore:
    """
    Stores and retrieves documents by semantic similarity using Pinecone.

    Each vector is upserted with:
      - id       : unique uuid string
      - values   : OpenAI embedding (1536-dim, L2-normalized)
      - metadata : {text, doc_type, **user_metadata}

    `namespace` maps directly to a Pinecone namespace, keeping brand
    knowledge bases and per-session conversation vectors isolated.
    """

    def __init__(self) -> None:
        self._index = _PineconeClient().index

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_documents(
        self,
        texts: List[str],
        namespace: str,
        doc_type: str = "kb",
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """
        Embed `texts` via OpenAI and upsert them into Pinecone under `namespace`.
        Returns the list of generated vector ids.
        """
        if not texts:
            return []

        metadatas = metadatas or [{} for _ in texts]
        matrix = embedding_provider.embed(texts)   # (n, 1536) L2-normalized

        ids = [str(uuid.uuid4()) for _ in texts]
        vectors = [
            {
                "id": vid,
                "values": matrix[i].tolist(),
                "metadata": {"text": text, "doc_type": doc_type, **meta},
            }
            for i, (vid, text, meta) in enumerate(zip(ids, texts, metadatas))
        ]

        self._index.upsert(vectors=vectors, namespace=namespace)
        logger.debug("Upserted %d vectors into namespace '%s'", len(vectors), namespace)
        return ids

    def add_document(
        self,
        text: str,
        namespace: str,
        doc_type: str = "kb",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Convenience wrapper for a single document."""
        ids = self.add_documents(
            texts=[text],
            namespace=namespace,
            doc_type=doc_type,
            metadatas=[metadata or {}],
        )
        return ids[0] if ids else ""

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def similarity_search(
        self,
        query: str,
        namespace: str,
        top_k: int = 5,
        doc_type: Optional[str] = None,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Return the `top_k` most similar documents to `query` in `namespace`.
        Each result has keys: text, metadata, score.
        """
        query_vec = embedding_provider.embed_one(query).tolist()

        pinecone_filter: Optional[Dict[str, Any]] = None
        if doc_type:
            pinecone_filter = {"doc_type": {"$eq": doc_type}}

        response = self._index.query(
            vector=query_vec,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True,
            filter=pinecone_filter,
        )

        results = []
        for match in response.get("matches", []):
            score = float(match.get("score", 0.0))
            if score < score_threshold:
                continue
            meta = dict(match.get("metadata", {}))
            results.append({
                "text": meta.pop("text", ""),
                "metadata": meta,
                "score": score,
            })

        return results

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_namespace(self, namespace: str) -> None:
        """Delete all vectors in `namespace`."""
        self._index.delete(delete_all=True, namespace=namespace)
        logger.debug("Deleted all vectors in namespace '%s'", namespace)

    def delete_by_ids(self, ids: List[str], namespace: str) -> None:
        """Delete specific vectors by their ids."""
        if ids:
            self._index.delete(ids=ids, namespace=namespace)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def count(self, namespace: str) -> int:
        """Return the number of vectors stored in `namespace`."""
        stats = self._index.describe_index_stats()
        ns_stats = stats.get("namespaces", {}).get(namespace, {})
        return int(ns_stats.get("vector_count", 0))
