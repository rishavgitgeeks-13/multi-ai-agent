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

    # Length in Additional Instructions (e.g. "10 words") must count as target too
    existing_instr = (state.get("additional_instructions") or "").strip()
    if existing_instr:
        try:
            from_extra = safety_service.extract_constraints(existing_instr)
            uc = dict(state.get("user_constraints") or {})
            # Prefer explicit length from additional instructions when present
            if from_extra.get("target_word_count"):
                uc["target_word_count"] = from_extra["target_word_count"]
                uc["word_count_flexible"] = from_extra.get(
                    "word_count_flexible", uc.get("word_count_flexible", True)
                )
                if from_extra.get("target_word_count_min") is not None:
                    uc["target_word_count_min"] = from_extra["target_word_count_min"]
                if from_extra.get("target_word_count_max") is not None:
                    uc["target_word_count_max"] = from_extra["target_word_count_max"]
                mentions = list(uc.get("raw_length_mentions") or [])
                mentions.extend(from_extra.get("raw_length_mentions") or [])
                uc["raw_length_mentions"] = mentions
            state["user_constraints"] = uc
        except Exception:
            pass

    state["brand_context"] = business_context_service.resolve(
        user_input=state["user_input"],
        brand=state.get("brand"),
    )

    # Language: manual UI selection wins; Auto follows prompt language (incl. Hinglish)
    try:
        from services.language_service import (
            language_writer_instruction,
            resolve_output_language,
        )

        out_lang, lang_source = resolve_output_language(
            state["user_input"],
            state.get("language"),
        )
        state["language"] = out_lang
        state["language_source"] = lang_source
        lang_line = language_writer_instruction(out_lang)
    except Exception:
        lang_line = ""
        out_lang = state.get("language") or "English"

    # Surface word-count + language + keep additional_instructions intact for Writer
    tw = (state.get("user_constraints") or {}).get("target_word_count")
    extras = []
    if tw:
        extras.append(
            f"User-requested target length: exactly about {tw} words. Adhere strictly."
        )
    if lang_line:
        extras.append(lang_line)
    # Preserve user additional instructions first (density, hashtags, etc.)
    merged = existing_instr
    if extras:
        block = "\n".join(extras)
        merged = f"{existing_instr}\n{block}".strip() if existing_instr else block
    state["additional_instructions"] = merged

    state["workflow_status"] = "RUNNING"
    state["next_agent"] = "research"
    logger.info(
        "manager_node PASS | topic=%s… | target_words=%s | language=%s",
        state["primary_topic"][:80],
        tw,
        out_lang,
    )
    return state
