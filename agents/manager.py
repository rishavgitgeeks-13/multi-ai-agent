"""
Manager Agent
=============

Entry point of the LangGraph workflow.

Responsibilities:
- Validate the incoming request.
- Resolve the business context.
- Initialize workflow state.
- Route execution to the Research Agent.

The Manager does NOT perform research, content generation,
or any LLM reasoning.

Note*: Instead of creating a class, I have used a node function, 
because LangGraph works more naturally with functions.
"""

from schemas.state import ContentState
from services.business_context_service import BusinessContextService


# Reusable service for resolving business-specific configuration.
business_context_service = BusinessContextService()


def manager_node(state: ContentState) -> ContentState:
    """Initialize the workflow and prepare the shared state."""

    # Validate user input before starting the workflow.
    if not state["user_input"].strip():
        raise ValueError("User input cannot be empty.")

    # Load the appropriate brand configuration for this request.
    state["brand_context"] = business_context_service.resolve(
        user_input=state["user_input"],
        brand=state.get("brand"),
    )

    # Initialize workflow tracking.
    state["workflow_status"] = "RUNNING"
    state["current_agent"] = "manager"
    state["next_agent"] = "research"

    return state