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


def manager_router(state: ContentState) -> str:
    """
    Route after the Manager Agent.

    BLOCKED → END (no research / writing)
    else    → "research"
    """
    if state.get("workflow_status") == "BLOCKED" or (state.get("safety") or {}).get(
        "blocked"
    ):
        logger.info(
            "Router manager → END | blocked category=%s",
            (state.get("safety") or {}).get("category", ""),
        )
        return END

    logger.info("Router manager → research")
    return "research"


def writer_router(state: ContentState) -> str:
    """
    Route after the Writer Agent.

    BLOCKED → END (draft discarded; skip Review)
    else    → "review"
    """
    if state.get("workflow_status") == "BLOCKED" or (state.get("safety") or {}).get(
        "blocked"
    ):
        logger.info("Router writer → END | blocked at writer")
        return END
    return "review"


def review_router(state: ContentState) -> str:
    """
    Route after the Review Agent.

    FAIL  (needs_revision=True)  → "writer"
    PASS / force-PASS            → "final_editor" (surgical QC)
    BLOCKED                      → END
    """
    if state.get("workflow_status") == "BLOCKED":
        logger.info("Router review → END | safety discarded draft")
        return END

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
        "Router → final_editor | score=%d | status=%s | below_target=%s",
        review.get("score", 0),
        review.get("status", ""),
        review.get("below_target", False),
    )
    return "final_editor"
