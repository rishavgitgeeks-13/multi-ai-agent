"""
Routing
=======

Conditional edge functions for the LangGraph workflow.

Each function receives the shared ContentState and returns a string
that LangGraph maps to the next node name (or END).
"""

import logging

from langgraph.graph import END

from schemas.state import ContentState
from config.settings import settings

logger = logging.getLogger(__name__)


def review_router(state: ContentState) -> str:
    """
    Route after the Review Agent.

    PASS  (needs_revision=False) → END
    FAIL  (needs_revision=True)  → "writer"

    The Review Agent itself enforces the max revision limit by setting
    needs_revision=False once the cap is reached, so this router only
    needs to inspect a single flag.
    """
    review = state.get("review", {})
    needs_revision = review.get("needs_revision", False)

    if needs_revision:
        revision_count = state.get("revision_count", 0)
        max_revisions = state.get("max_revision_count", settings.MAX_REVIEW_ITERATIONS)
        logger.info(
            "Router → writer | revision %d / %d | score=%d",
            revision_count,
            max_revisions,
            review.get("score", 0),
        )
        return "writer"

    logger.info(
        "Router → END | score=%d | status=%s",
        review.get("score", 0),
        review.get("status", ""),
    )
    return END
