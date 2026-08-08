"""
Review Agent
============

Evaluates the content draft produced by the Writer Agent.

Responsibilities:
    - Safety fallback: discard any abusive / off-policy / inverted draft
    - Run ReviewService to score the draft across five dimensions
    - Enforce the maximum revision limit (settings.MAX_REVIEW_ITERATIONS)
    - If PASS  → mark workflow COMPLETED, route to END
    - If FAIL  → increment revision_count, inject rewrite_instruction
                 into strategy, route back to the Writer Agent
    - Update shared state with the review result

The Review Agent does not generate or modify content (except discarding
unsafe output).
"""

import logging

from schemas.state import ContentState
from services.review_service import ReviewService, PASS_THRESHOLD
from services.safety_service import safety_service

from config.settings import settings

logger = logging.getLogger(__name__)

review_service = ReviewService()


def review_node(state: ContentState) -> ContentState:
    """Evaluate the draft and decide: discard, revise, or complete."""
    logger.info(
        "review_node() | revision_count=%d | max=%d",
        state.get("revision_count", 0),
        state.get("max_revision_count", settings.MAX_REVIEW_ITERATIONS),
    )

    draft = state.get("draft") or ""

    # ------------------------------------------------------------------
    # Safety fallback — discard completely if draft is unsafe / inverted
    # ------------------------------------------------------------------
    draft_safety = safety_service.evaluate_draft(
        draft,
        primary_topic=state.get("primary_topic") or "",
        user_input=state.get("user_input") or "",
        request_id=state.get("request_id", ""),
        session_id=state.get("session_id", ""),
        brand=state.get("brand"),
        content_type=state.get("content_type", ""),
        source="review",
    )
    if draft_safety.get("blocked"):
        msg = draft_safety.get("message") or (
            "Generated content was discarded because it violated content policy."
        )
        state["draft"] = ""
        state["metadata"] = {}
        state["formatted_output"] = {}
        state["final_output"] = {}
        state["workflow_status"] = "BLOCKED"
        state["current_agent"] = "review"
        state["next_agent"] = "end"
        state["errors"] = list(state.get("errors") or []) + [msg]
        state["safety"] = {
            **(state.get("safety") or {}),
            "allowed": False,
            "blocked": True,
            "category": draft_safety.get("category", "draft_blocked"),
            "reason": draft_safety.get("reason", ""),
            "message": msg,
            "discarded_at": "review",
        }
        state["review"] = {
            "score": 0,
            "status": "BLOCKED",
            "needs_revision": False,
            "feedback": [],
            "issues": [draft_safety.get("reason") or "Policy violation in draft"],
            "rewrite_instruction": "",
            "dimension_scores": {},
            "revision_number": state.get("revision_count", 0),
        }
        logger.warning(
            "review_node DISCARDED draft | category=%s | reason=%s",
            draft_safety.get("category"),
            draft_safety.get("reason"),
        )
        return state

    strategy = state["strategy"]
    brand_context = state["brand_context"]
    revision_count = state.get("revision_count", 0)
    max_revisions = state.get("max_revision_count", settings.MAX_REVIEW_ITERATIONS)

    # ------------------------------------------------------------------
    # Run the quality review
    # ------------------------------------------------------------------
    review = review_service.run(
        draft=draft,
        strategy=strategy,
        brand_context=brand_context,
        revision_count=revision_count,
        primary_topic=state.get("primary_topic") or "",
        user_input=state.get("user_input") or "",
    )

    # Word-count adherence when user requested a target
    constraints = state.get("user_constraints") or {}
    target = constraints.get("target_word_count")
    length_ok = True
    if target:
        actual = len(draft.split())
        flexible = constraints.get("word_count_flexible", True)
        tmin = constraints.get("target_word_count_min")
        tmax = constraints.get("target_word_count_max")
        # Prefer explicit range from prompt (e.g. 700-800 words)
        if tmin is not None and tmax is not None:
            lo, hi = int(tmin), int(tmax)
        elif int(target) <= 50:
            lo, hi = max(1, int(target) - 2), int(target) + 2
        elif flexible:
            lo, hi = int(target * 0.85), int(target * 1.15)
        else:
            lo, hi = int(target * 0.95), int(target * 1.05)
        if actual < lo or actual > hi:
            length_ok = False
            review["issues"] = list(review.get("issues") or [])
            review["issues"].append(
                f"Word count {actual} is outside the user-requested target "
                f"of ~{target} words (acceptable {lo}-{hi})."
            )
            if review.get("needs_revision") or revision_count < max_revisions:
                review["needs_revision"] = True
                review["status"] = "FAIL"
                review["rewrite_instruction"] = (
                    (review.get("rewrite_instruction") or "").strip()
                    + f"\nAdjust length to approximately {target} words "
                    f"(current ~{actual}). Do not expand into a long article. "
                    f"Do not change the primary topic."
                ).strip()

    # Topic fidelity reminder in rewrite instructions
    primary = (state.get("primary_topic") or "").strip()
    if review.get("needs_revision") and primary:
        review["rewrite_instruction"] = (
            (review.get("rewrite_instruction") or "").strip()
            + f"\nStay strictly on this primary topic (do not invert roles or change subject): {primary}"
        ).strip()

    # ------------------------------------------------------------------
    # Enforce maximum revision limit
    # ------------------------------------------------------------------
    word_count = len(draft.split())
    content_type = (strategy.get("content_type") or state.get("content_type") or "").lower()
    # Default article floor only when the user did NOT set a length.
    if target:
        min_words = max(1, int(target) // 2) if int(target) > 50 else 1
    else:
        min_words = (
            settings.MIN_ARTICLE_WORDS
            if content_type in ("blog", "article")
            else 1
        )

    if review["needs_revision"] and revision_count >= max_revisions:
        # Never force-pass empty / far-too-short content — mark FAILED instead.
        if word_count < max(1, min_words // 4 if not target else min_words):
            logger.error(
                "Max revisions reached with unusable draft (%d words) — marking FAILED",
                word_count,
            )
            review["needs_revision"] = False
            review["status"] = "FAIL"
            review["issues"] = list(review.get("issues") or []) + [
                f"Generation failed after {max_revisions} revisions: "
                f"content only {word_count} words (unusable)."
            ]
            state["workflow_status"] = "FAILED"
            state["errors"] = list(state.get("errors") or []) + [
                f"Content generation failed: draft too short ({word_count} words) "
                f"after maximum revisions. Please retry."
            ]
            state["current_agent"] = "review"
            state["next_agent"] = "end"
            state["review"] = review
            return state

        # Do not force-PASS when the user length ask is still violated.
        if target and not length_ok:
            logger.warning(
                "Max revisions reached with length miss (target=%s, actual=%d) — keeping FAIL",
                target,
                word_count,
            )
            review["needs_revision"] = False
            review["status"] = "FAIL"
            review["feedback"] = list(review.get("feedback") or []) + [
                f"Maximum revision limit ({max_revisions}) reached without hitting "
                f"the requested ~{target}-word length (got {word_count})."
            ]
            state["workflow_status"] = "COMPLETED"
            state["current_agent"] = "review"
            state["next_agent"] = "end"
            state["review"] = review
            return state

        # Comments: never force-PASS an off-format reply (pitch / incomplete / hashtags).
        platform = (strategy.get("platform") or state.get("platform") or "").lower()
        if content_type == "comment" or platform == "comment":
            hard_issues = [
                i
                for i in (review.get("issues") or [])
                if any(
                    k in str(i).lower()
                    for k in (
                        "outreach",
                        "sales",
                        "hashtag",
                        "incomplete",
                        "cut off",
                        "outside the requested",
                        "too long",
                    )
                )
            ]
            if hard_issues:
                logger.warning(
                    "Max revisions reached on comment with format issues — keeping FAIL"
                )
                review["needs_revision"] = False
                review["status"] = "FAIL"
                review["feedback"] = list(review.get("feedback") or []) + [
                    f"Maximum revision limit ({max_revisions}) reached; "
                    "comment still off-format (not force-passed)."
                ]
                state["workflow_status"] = "COMPLETED"
                state["current_agent"] = "review"
                state["next_agent"] = "end"
                state["review"] = review
                return state

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
        state["strategy"]["rewrite_instruction"] = review["rewrite_instruction"]
        state["current_agent"] = "review"
        state["next_agent"] = "writer"
        logger.info(
            "Review FAIL | score=%d | sending back to writer (revision %d of %d)",
            review["score"],
            state["revision_count"],
            max_revisions,
        )
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
