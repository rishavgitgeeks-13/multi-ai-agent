"""
Writer Agent
============

Generates content using the strategy blueprint.

Responsibilities:
- Read the strategy blueprint.
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


writer_service = WriterService()
metadata_service = MetadataService()
formatter = Formatter()
json_builder = JSONBuilder()


def writer_node(state: ContentState) -> ContentState:
    """Generate content from the strategy blueprint."""

    # Generate the first draft.
    draft = writer_service.run(
        user_input=state["user_input"],
        research_data=state["research_data"],
        strategy=state["strategy"],
        brand_context=state["brand_context"],
    )

    # Generate metadata.
    metadata = metadata_service.run(
        draft=draft,
        strategy=state["strategy"],
    )

    # Format the final draft.
    formatted_output = formatter.run(
        draft=draft,
        strategy=state["strategy"],
    )

    # Build the response object.
    final_output = json_builder.run(
        content=formatted_output,
        metadata=metadata,
        strategy=state["strategy"],
    )

    # Update workflow state.
    state["draft"] = draft
    state["metadata"] = metadata
    state["formatted_output"] = formatted_output
    state["final_output"] = final_output

    state["current_agent"] = "writer"
    state["next_agent"] = "review"

    return state