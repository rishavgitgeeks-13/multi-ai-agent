"""
LangGraph Workflow

Defines the execution flow for the multi-agent content generation system.

Workflow:
START
    ↓
Manager
    ↓
Research
    ↓
Strategy
    ↓
Writer
    ↓
Review
    ├── PASS → END
    └── FAIL → Writer
"""

from langgraph.graph import StateGraph, START, END

from schemas.state import ContentState

from agents.manager import manager_node
from agents.research import research_node
from agents.strategy import strategy_node
from agents.writer import writer_node
from agents.review import review_node
from graphs.routing import review_router


builder = StateGraph(ContentState)

# Register workflow nodes.
builder.add_node("manager", manager_node)
builder.add_node("research", research_node)
builder.add_node("strategy", strategy_node)
builder.add_node("writer", writer_node)
builder.add_node("review", review_node)

# Define workflow execution.
builder.add_edge(START, "manager")
builder.add_edge("manager", "research")
builder.add_edge("research", "strategy")
builder.add_edge("strategy", "writer")
builder.add_edge("writer", "review")

# Review decides whether to finish or rewrite.
builder.add_conditional_edges(
    "review",
    review_router,
    {
        "writer": "writer",
        END: END,
    },
)

def create_graph():
    return builder.compile()


graph = create_graph()