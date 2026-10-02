"""
Research Agent
==============

Collects relevant information for the user's request.

Flow (with Strategy + Writer):
  1. Research — fetch on-brief evidence (BriefLock markets/topic)
  2. Strategy — plan outline FROM that research
  3. Writer  — write proper content from the plan + research

Responsibilities:
- Read the user query, brand context, and brief_lock.
- Invoke the Research Service (multi-query fan-out).
- Store the research results in the shared state.
- Route the workflow to the Strategy Agent.

The Research Agent does not perform keyword analysis,
content generation, or business decisions.
"""

import logging

from schemas.state import ContentState
from services.research_service import ResearchService

logger = logging.getLogger(__name__)

# Reusable research service instance.
research_service = ResearchService()


def research_node(state: ContentState) -> ContentState:
    """Execute the research stage of the workflow."""

    query = state.get("primary_topic") or state["user_input"]
    brand_context = dict(state.get("brand_context") or {})
    # Ensure BriefLock rides with research for on-brief query fan-out.
    if state.get("brief_lock") and not brand_context.get("brief_lock"):
        brand_context["brief_lock"] = state["brief_lock"]

    logger.info(
        "research_node | query=%s… | brand=%s | markets=%s",
        (query or "")[:80],
        brand_context.get("namespace"),
        (brand_context.get("brief_lock") or {}).get("markets"),
    )

    research_result = research_service.run(
        query=query,
        brand_context=brand_context,
    )

    state["research_data"] = research_result
    state["retrieved_documents"] = research_result.get("documents", [])
    state["sources"] = research_result.get("sources", [])
    state["brand_context"] = brand_context

    state["current_agent"] = "research"
    state["next_agent"] = "strategy"

    logger.info(
        "research_node done | docs=%d | sources=%d | stats=%d | incidents=%d",
        len(state["retrieved_documents"] or []),
        len(state["sources"] or []),
        len(research_result.get("statistics") or []),
        len(research_result.get("incidents") or []),
    )

    return state
