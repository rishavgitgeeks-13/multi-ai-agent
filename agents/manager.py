"""
Manager Agent
=============

Entry point of the LangGraph workflow.

Responsibilities:
- Validate the incoming request.
- Run the content safety / policy gate (hard stop on abuse).
- Extract user constraints (e.g. target word count).
- Lock the primary topic for downstream agents.
- Resolve the business context.
- Route to Research (pass) or END (blocked).
"""

import logging

from schemas.state import ContentState
from services.business_context_service import BusinessContextService
from services.safety_service import safety_service

logger = logging.getLogger(__name__)

business_context_service = BusinessContextService()


def manager_node(state: ContentState) -> ContentState:
    """Initialize the workflow, enforce policy, and prepare shared state."""

    if not state["user_input"].strip():
        raise ValueError("User input cannot be empty.")

    decision = safety_service.evaluate_request(
        state["user_input"],
        request_id=state.get("request_id", ""),
        session_id=state.get("session_id", ""),
        brand=state.get("brand"),
        content_type=state.get("content_type", ""),
        source="manager",
    )

    state["primary_topic"] = decision.get("primary_topic") or state["user_input"]
    state["user_constraints"] = decision.get("user_constraints") or {}
    state["safety"] = {
        "allowed": decision.get("allowed", True),
        "blocked": decision.get("blocked", False),
        "category": decision.get("category", ""),
        "reason": decision.get("reason", ""),
        "message": decision.get("message", ""),
        "defensive_allow": decision.get("defensive_allow", False),
    }
    state["current_agent"] = "manager"

    if decision.get("blocked"):
        msg = decision.get("message") or "Request blocked by content policy."
        state["workflow_status"] = "BLOCKED"
        state["next_agent"] = "end"
        state["draft"] = ""
        state["final_output"] = {}
        state["metadata"] = {}
        state["errors"] = list(state.get("errors") or []) + [msg]
        logger.warning(
            "manager_node BLOCKED | category=%s | reason=%s",
            decision.get("category"),
            decision.get("reason"),
        )
        return state

    state["brand_context"] = business_context_service.resolve(
        user_input=state["user_input"],
        brand=state.get("brand"),
    )

    # Surface word-count constraint for Writer via strategy later
    tw = (state.get("user_constraints") or {}).get("target_word_count")
    if tw:
        extra = f"User-requested target length: exactly about {tw} words. Adhere strictly."
        existing = (state.get("additional_instructions") or "").strip()
        state["additional_instructions"] = (
            f"{existing}\n{extra}".strip() if existing else extra
        )

    state["workflow_status"] = "RUNNING"
    state["next_agent"] = "research"
    logger.info(
        "manager_node PASS | topic=%s… | target_words=%s",
        state["primary_topic"][:80],
        tw,
    )
    return state
