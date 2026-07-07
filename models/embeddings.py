"""
Centralized embedding provider for the Editorial Intelligence System.

Model : OpenAI text-embedding-3-small  (1536-dimensional)

Consumed by
-----------
  SEOService (services/seo_service.py)
    - Step 5 : semantic_similarity  — keyword vs. user query
    - Step 6 : pain_point           — keyword vs. brand pain points
    - Step 7 : brand_relevance      — keyword vs. keyword direction

  VectorStore (memory/vector.py)
    - Pinecone upsert  — embed documents before storing
    - Pinecone query   — embed the search query before retrieval

All returned arrays are L2-normalized so that:
    np.dot(a, b) == cosine_similarity(a, b)

Import pattern
--------------
    from models.embeddings import embedding_provider

    matrix  = embedding_provider.embed(["text a", "text b"])   # (2, 1536)
    vector  = embedding_provider.embed_one("text a")            # (1536,)
    score   = float(np.dot(vec_a, vec_b))                       # cosine similarity
"""

import logging
from typing import List

import numpy as np
from openai import OpenAI

from config.settings import settings

logger = logging.getLogger(__name__)


class EmbeddingProvider:
    """
    Singleton wrapper around the OpenAI Embeddings API.

    All vectors are L2-normalized on return so callers can compute
    cosine similarity with a plain dot product (no extra math needed).
    """

    _instance: "EmbeddingProvider" = None
    _client: OpenAI = None

    def __new__(cls) -> "EmbeddingProvider":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._client = OpenAI(api_key=settings.OPENAI_API_KEY)
            logger.info(
                "EmbeddingProvider initialized | model=%s | dim=%d",
                settings.OPENAI_EMBEDDING_MODEL,
                settings.OPENAI_EMBEDDING_DIMENSION,
            )
        return cls._instance

    # ------------------------------------------------------------------
    # Core embedding methods
    # ------------------------------------------------------------------

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of strings.

        Parameters
        ----------
        texts : List[str]
            Non-empty list of strings to embed.

        Returns
        -------
        np.ndarray  shape (len(texts), OPENAI_EMBEDDING_DIMENSION)
            L2-normalized — each row has unit length.
            dot_product(row_i, row_j) == cosine_similarity(row_i, row_j)
        """
        if not texts:
            return np.empty((0, settings.OPENAI_EMBEDDING_DIMENSION), dtype=float)

        response = self._client.embeddings.create(
            input=texts,
            model=settings.OPENAI_EMBEDDING_MODEL,
        )
        vectors = np.array(
            [item.embedding for item in response.data], dtype=float
        )
        return self._l2_normalize(vectors)

    def embed_one(self, text: str) -> np.ndarray:
        """
        Embed a single string.

        Returns
        -------
        np.ndarray  shape (OPENAI_EMBEDDING_DIMENSION,)
            Unit-length vector.
        """
        return self.embed([text])[0]

    # ------------------------------------------------------------------
    # Similarity helpers
    # ------------------------------------------------------------------

    def semantic_similarity(
        self,
        query: str,
        candidates: List[str],
    ) -> np.ndarray:
        """
        Compute cosine similarity between `query` and each string in `candidates`.

        Returns
        -------
        np.ndarray  shape (len(candidates),)
            Values in [0, 1]. Higher = more semantically similar.

        Usage in SEO pipeline
        ----------------------
            sims = embedding_provider.semantic_similarity(
                query=user_input,
                candidates=[kw.keyword for kw in keyword_candidates],
            )
            for candidate, score in zip(keyword_candidates, sims):
                candidate.scores["semantic_similarity"] = float(score)
        """
        query_vec = self.embed_one(query)           # (dim,)
        candidate_matrix = self.embed(candidates)   # (n, dim)
        sims = np.clip(candidate_matrix @ query_vec, 0.0, 1.0)
        return sims

    def similarity_matrix(
        self,
        texts_a: List[str],
        texts_b: List[str],
    ) -> np.ndarray:
        """
        Compute an (m, n) cosine similarity matrix between two lists.

        Returns
        -------
        np.ndarray  shape (len(texts_a), len(texts_b))
        """
        matrix_a = self.embed(texts_a)   # (m, dim)
        matrix_b = self.embed(texts_b)   # (n, dim)
        return np.clip(matrix_a @ matrix_b.T, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
        """Normalize each row to unit length in-place safe."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        return vectors / norms


# ==========================================================================
# Module-level singleton — import this directly
# ==========================================================================

embedding_provider = EmbeddingProvider()
