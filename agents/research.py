"""
Research Agent
==============

Collects relevant information for the user's request.

Responsibilities:
- Read the user query and brand context.
- Invoke the Research Service.
- Store the research results in the shared state.
- Route the workflow to the Strategy Agent.

The Research Agent does not perform keyword analysis,
content generation, or business decisions.
"""

from schemas.state import ContentState
from services.research_service import ResearchService


# Reusable research service instance.
research_service = ResearchService()


def research_node(state: ContentState) -> ContentState:
    """Execute the research stage of the workflow."""

    # Retrieve research from internal and external sources.
    research_result = research_service.run(
        query=state.get("primary_topic") or state["user_input"],
        brand_context=state["brand_context"],
    )

    # Update the shared state with research results.
    state["research_data"] = research_result
    state["retrieved_documents"] = research_result.get("documents", [])
    state["sources"] = research_result.get("sources", [])

    # Update workflow tracking.
    state["current_agent"] = "research"
    state["next_agent"] = "strategy"

    return state
