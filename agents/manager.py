"""
Manager Agent
=============

Entry point of the LangGraph workflow.

Responsibilities:
- Validate the incoming request.
- Run the content safety / policy gate (hard stop on abuse).
- Extract user constraints (e.g. target word count).
- Apply format hints from Extra tips (LinkedIn / email / etc.).
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

# When Extra tips ask for a platform format, inject hard writer rules.
_FORMAT_RULES = {
    "linkedin": (
        "LINKEDIN POST FORMAT (mandatory from user notes):\n"
        "- Write a LinkedIn post — not a website article or SEO blog.\n"
        "- Open with a strong creative hook in the first 1–2 lines.\n"
        "- Short paragraphs (1–3 lines). Prefer line breaks over dense ## essay sections.\n"
        "- Professional content-writer voice; keep it scannable for mobile.\n"
        "- If a word count is specified (e.g. 1000 words), you MUST hit that length "
        "with real substance (stories, examples, steps) — do not stop at ~600 words.\n"
        "- End with a light engagement CTA or question when natural.\n"
    ),
    "instagram": (
        "INSTAGRAM CAPTION FORMAT (mandatory from user notes):\n"
        "- Short, scannable caption. Hook first. Hashtags at the end only.\n"
    ),
    "x": (
        "X / TWITTER FORMAT (mandatory from user notes):\n"
        "- Keep it punchy and platform-native. Respect short length.\n"
    ),
    "email": (
        "EMAIL FORMAT (mandatory from user notes):\n"
        "- Subject-ready opener, clear body, one CTA.\n"
    ),
}


def _format_hint_from_notes(text: str) -> str:
    """Detect platform/format requests buried in Extra tips or the prompt."""
    t = (text or "").lower()
    if not t:
        return ""
    if any(x in t for x in ("linkedin", "linked in", "li post")):
        return "linkedin"
    if "instagram" in t or "ig caption" in t:
        return "instagram"
    if any(x in t for x in ("twitter", "tweet", "x post", "x.com")):
        return "x"
    if any(x in t for x in ("email", "newsletter", "cold mail")):
        return "email"
    if "carousel" in t:
        return "carousel"
    return ""


def _apply_format_override(state: ContentState, hint: str) -> None:
    """Align content_type / platform / brand_context with Extra tips."""
    if not hint:
        return
    bc = dict(state.get("brand_context") or {})
    if hint == "linkedin":
        state["content_type"] = "linkedin"
        state["platform"] = "linkedin"
        bc["content_type"] = "linkedin"
        bc["platform"] = "linkedin"
        bc["workflow"] = "social"
    elif hint == "instagram":
        state["content_type"] = "linkedin"
        state["platform"] = "instagram"
        bc["content_type"] = "linkedin"
        bc["platform"] = "instagram"
        bc["workflow"] = "social"
    elif hint == "x":
        state["content_type"] = "linkedin"
        state["platform"] = "x"
        bc["content_type"] = "linkedin"
        bc["platform"] = "x"
        bc["workflow"] = "social"
    elif hint == "carousel":
        state["content_type"] = "carousel"
        state["platform"] = "carousel"
        bc["content_type"] = "carousel"
        bc["platform"] = "carousel"
        bc["workflow"] = "social"
    elif hint == "email":
        state["content_type"] = "email"
        state["platform"] = "email"
        bc["content_type"] = "email"
        bc["platform"] = "email"
        bc["workflow"] = "email"
        bc["campaign_type"] = bc.get("campaign_type") or "newsletter"
    state["brand_context"] = bc
    logger.info(
        "manager format override from notes | hint=%s | content_type=%s | platform=%s",
        hint,
        state.get("content_type"),
        state.get("platform"),
    )


def manager_node(state: ContentState) -> ContentState:
    """Initialize the workflow, enforce policy, and prepare shared state."""

    if not state["user_input"].strip():
        raise ValueError("User input cannot be empty.")

    decision = safety_service.evaluate_request(
        state["user_input"],
        additional_instructions=state.get("additional_instructions") or "",
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

    # Length + format in Additional Instructions must count
    existing_instr = (state.get("additional_instructions") or "").strip()
    combined_for_hints = f"{state.get('user_input') or ''}\n{existing_instr}".strip()

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

    # Resolve brand using topic + notes so "linkedin" in Extra tips is seen
    state["brand_context"] = business_context_service.resolve(
        user_input=combined_for_hints,
        brand=state.get("brand"),
    )

    # Hard override from notes (works even if Content workflow locked article)
    format_hint = _format_hint_from_notes(combined_for_hints)
    if format_hint:
        _apply_format_override(state, format_hint)

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

    # Surface word-count + format + language; keep user notes first
    tw = (state.get("user_constraints") or {}).get("target_word_count")
    flexible = bool(
        (state.get("user_constraints") or {}).get("word_count_flexible", True)
    )
    extras = []
    if format_hint and format_hint in _FORMAT_RULES:
        extras.append(_FORMAT_RULES[format_hint])
    if tw:
        if flexible:
            extras.append(
                f"User-requested target length: about {tw} words "
                f"(stay within ~15% of {tw}). Do not pad past the upper band."
            )
        else:
            extras.append(
                f"User-requested target length: HARD LIMIT ~{tw} words. "
                f"Stay inside {int(tw * 0.95)}–{int(tw * 1.05)} words. "
                "Stop writing when you reach the target — do not pad with filler sections."
            )
    if lang_line:
        extras.append(lang_line)

    merged = existing_instr
    if extras:
        block = "\n".join(extras)
        merged = f"{existing_instr}\n{block}".strip() if existing_instr else block
    state["additional_instructions"] = merged

    state["workflow_status"] = "RUNNING"
    state["next_agent"] = "research"
    logger.info(
        "manager_node PASS | topic=%s… | target_words=%s | flexible=%s | format=%s | language=%s",
        state["primary_topic"][:80],
        tw,
        flexible,
        format_hint or state.get("content_type"),
        out_lang,
    )
    return state
