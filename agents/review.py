"""
Review Agent
============

Evaluates the content draft produced by the Writer Agent.

Responsibilities:
    - Run ReviewService to score the draft across five dimensions
    - Enforce the maximum revision limit (settings.MAX_REVIEW_ITERATIONS)
    - If PASS  → mark workflow COMPLETED, route to END
    - If FAIL  → increment revision_count, inject rewrite_instruction
                 into strategy, route back to the Writer Agent
    - Update shared state with the review result

The Review Agent does not generate or modify content.
"""

import logging

from schemas.state import ContentState
from services.review_service import ReviewService, PASS_THRESHOLD

from config.settings import settings

logger = logging.getLogger(__name__)

review_service = ReviewService()


def review_node(state: ContentState) -> ContentState:
    """Evaluate the draft and decide: revise or complete."""
    logger.info(
        "review_node() | revision_count=%d | max=%d",
        state.get("revision_count", 0),
        state.get("max_revision_count", settings.MAX_REVIEW_ITERATIONS),
    )

    draft = state["draft"]
    strategy = state["strategy"]
    brand_context = state["brand_context"]
    revision_count = state.get("revision_count", 0)
    max_revisions = state.get("max_revision_count", settings.MAX_REVIEW_ITERATIONS)

    # ------------------------------------------------------------------
    # Run the review
    # ------------------------------------------------------------------
    review = review_service.run(
        draft=draft,
        strategy=strategy,
        brand_context=brand_context,
        revision_count=revision_count,
    )

    # ------------------------------------------------------------------
    # Enforce maximum revision limit
    # ------------------------------------------------------------------
    if review["needs_revision"] and revision_count >= max_revisions:
        logger.warning(
            "Max revision limit (%d) reached — forcing PASS with score %d",
            max_revisions,
            review["score"],
        )
        review["needs_revision"] = False
        review["status"] = "PASS"
        review["feedback"].append(
            f"Maximum revision limit ({max_revisions}) reached. "
            f"Force-passed at score {review['score']} "
            f"(quality target is {PASS_THRESHOLD}+). "
            f"Treat as below target if score < {PASS_THRESHOLD}."
        )

    # ------------------------------------------------------------------
    # Route: FAIL → inject instruction and send back to Writer
    # ------------------------------------------------------------------
    if review["needs_revision"]:
        state["revision_count"] = revision_count + 1

        # Inject the rewrite instruction into strategy so WriterService
        # can adapt its prompts on the next run.
        state["strategy"]["rewrite_instruction"] = review["rewrite_instruction"]

        state["current_agent"] = "review"
        state["next_agent"] = "writer"

        logger.info(
            "Review FAIL | score=%d | sending back to writer (revision %d of %d)",
            review["score"],
            state["revision_count"],
            max_revisions,
        )

    # ------------------------------------------------------------------
    # Route: PASS → complete the workflow
    # ------------------------------------------------------------------
    else:
        state["workflow_status"] = "COMPLETED"
        state["current_agent"] = "review"
        state["next_agent"] = "end"

        logger.info(
            "Review PASS | score=%d | workflow complete",
            review["score"],
        )

    state["review"] = review
    return state
