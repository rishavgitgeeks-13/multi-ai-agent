"""
Writer Agent
============

Generates content using the strategy blueprint.

Responsibilities:
- Read the strategy blueprint and locked primary topic.
- Honour user word-count constraints when provided.
- Generate the content.
- Format the output.
- Generate metadata.
- Build the final response.
- Route the workflow to the Review Agent.

The Writer Agent does not perform research,
SEO planning, keyword ranking, or strategy generation.
"""

from schemas.state import ContentState

from services.metadata_service import MetadataService
from services.formatter import Formatter
from services.json_builder import JSONBuilder
from services.writer_service import WriterService
from services.safety_service import safety_service


writer_service = WriterService()
metadata_service = MetadataService()
formatter = Formatter()
json_builder = JSONBuilder()


def writer_node(state: ContentState) -> ContentState:
    """Generate content from the strategy blueprint."""

    strategy = dict(state["strategy"] or {})
    previous_draft = ""
    if str(strategy.get("rewrite_instruction", "")).strip():
        previous_draft = state.get("draft") or ""

    primary_topic = (state.get("primary_topic") or "").strip()
    if primary_topic:
        strategy["primary_topic"] = primary_topic

    constraints = state.get("user_constraints") or {}
    if constraints.get("target_word_count"):
        strategy["target_word_count"] = int(constraints["target_word_count"])
        strategy["word_count_flexible"] = bool(
            constraints.get("word_count_flexible", True)
        )

    # Generate the first draft (or surgically revise on FAIL → rewrite).
    draft = writer_service.run(
        user_input=state["user_input"],
        research_data=state["research_data"],
        strategy=strategy,
        brand_context=state["brand_context"],
        previous_draft=previous_draft,
        primary_topic=primary_topic,
        additional_instructions=state.get("additional_instructions") or "",
    )

    # Mid-pipeline safety scan — if Writer produced blocked content, clear it
    # so Review's discard path is deterministic and nothing leaks downstream.
    draft_safety = safety_service.evaluate_draft(
        draft,
        primary_topic=primary_topic,
        user_input=state.get("user_input") or "",
        request_id=state.get("request_id", ""),
        session_id=state.get("session_id", ""),
        brand=state.get("brand"),
        content_type=state.get("content_type", ""),
        source="writer",
    )
    if draft_safety.get("blocked"):
        state["draft"] = ""
        state["metadata"] = {}
        state["formatted_output"] = {}
        state["final_output"] = {}
        state["workflow_status"] = "BLOCKED"
        state["current_agent"] = "writer"
        state["next_agent"] = "end"
        msg = draft_safety.get("message") or (
            "Generated content was blocked by content policy."
        )
        state["errors"] = list(state.get("errors") or []) + [msg]
        state["safety"] = {
            **(state.get("safety") or {}),
            "allowed": False,
            "blocked": True,
            "category": draft_safety.get("category", "writer_blocked"),
            "reason": draft_safety.get("reason", ""),
            "message": msg,
            "discarded_at": "writer",
        }
        return state

    metadata = metadata_service.run(
        draft=draft,
        strategy=strategy,
    )

    formatted_output = formatter.run(
        draft=draft,
        strategy=strategy,
    )

    final_output = json_builder.run(
        content=formatted_output,
        metadata=metadata,
        strategy=strategy,
    )

    state["draft"] = draft
    state["metadata"] = metadata
    state["formatted_output"] = formatted_output
    state["final_output"] = final_output

    state["current_agent"] = "writer"
    state["next_agent"] = "review"

    return state
