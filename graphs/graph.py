"""
LangGraph Workflow

Defines the execution flow for the multi-agent content generation system.

Workflow:
START
    ↓
Manager
    ├── BLOCKED → END
    └── PASS
         ↓
      Research
         ↓
      Strategy
         ↓
      Writer
         ↓
      Review
         ├── QUALITY FAIL → Writer
         ├── SAFETY FAIL → END
         └── PASS / force-PASS → Final Editor → END
"""

from langgraph.graph import StateGraph, START, END

from schemas.state import ContentState

from agents.manager import manager_node
from agents.research import research_node
from agents.strategy import strategy_node
from agents.writer import writer_node
from agents.review import review_node
from agents.final_editor import final_editor_node
from graphs.routing import manager_router, writer_router, review_router


builder = StateGraph(ContentState)

# Register workflow nodes.
builder.add_node("manager", manager_node)
builder.add_node("research", research_node)
builder.add_node("strategy", strategy_node)
builder.add_node("writer", writer_node)
builder.add_node("review", review_node)
builder.add_node("final_editor", final_editor_node)

# Define workflow execution.
builder.add_edge(START, "manager")
builder.add_conditional_edges(
    "manager",
    manager_router,
    {
        "research": "research",
        END: END,
    },
)
builder.add_edge("research", "strategy")
builder.add_edge("strategy", "writer")
builder.add_conditional_edges(
    "writer",
    writer_router,
    {
        "review": "review",
        END: END,
    },
)

# Review: rewrite loop OR surgical final edit then finish.
builder.add_conditional_edges(
    "review",
    review_router,
    {
        "writer": "writer",
        "final_editor": "final_editor",
        END: END,
    },
)
builder.add_edge("final_editor", END)


def create_graph():
    return builder.compile()


graph = create_graph()
