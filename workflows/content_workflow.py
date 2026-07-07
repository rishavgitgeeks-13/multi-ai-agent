"""
Content Workflow
================

Orchestrates the full 5-agent pipeline for long-form content:
  - Article  (~2200 words, SEO-optimised, website)
  - Blog     (~1800 words, conversational, website)

Wraps the core LangGraph graph from `graphs.graph` with:
  - Content-type defaults (content_type, platform, objective)
  - Optional ConversationMemory integration (pass session_id)
  - Input validation before graph entry
  - Structured return dict with content + review summary + errors

Usage
-----
    from workflows.content_workflow import ContentWorkflow

    workflow = ContentWorkflow()

    result = workflow.run(
        user_input="How AI agents are transforming SMB operations",
        content_type="article",         # "article" | "blog"
        brand="Futuristix",             # optional brand hint
        objective="seo",
        session_id="sess-abc123",       # enables ConversationMemory
    )

    if result["ok"]:
        markdown = result["final_output"]["content"]["markdown"]
        seo      = result["final_output"]["seo"]
    else:
        print(result["errors"])
"""

import logging
import uuid
from typing import Any, Dict, List, Optional

from config.settings import settings
from graphs.graph import graph

logger = logging.getLogger(__name__)

# Valid values for this workflow
_VALID_CONTENT_TYPES = {"article", "blog"}
_VALID_OBJECTIVES = {"seo", "authority", "engagement", "leads"}


class ContentWorkflow:
    """
    Long-form content workflow: article and blog post generation.

    The workflow calls the 5-agent pipeline (Manager → Research → Strategy
    → Writer → Review) with content-type defaults pre-filled so callers
    don't need to know internal state field names.
    """

    def __init__(self) -> None:
        self._graph = graph

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        user_input: str,
        content_type: str = "article",
        brand: Optional[str] = None,
        objective: str = "seo",
        language: str = "English",
        additional_instructions: str = "",
        session_id: Optional[str] = None,
        max_revisions: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run the long-form content workflow.

        Parameters
        ----------
        user_input            : Topic, question, or brief for the article.
        content_type          : "article" (default, ~2200 words) | "blog" (~1800 words).
        brand                 : Optional brand name or alias passed as a hint in
                                user_input. The Manager Agent resolves it via
                                BusinessContextService; passing the hint here
                                prepends it to the query so resolution is reliable.
        objective             : "seo" (default) | "authority" | "engagement" | "leads".
        language              : "English" (default) | "Hindi".
        additional_instructions: Free-text modifier appended to writer prompts.
        session_id            : If provided, saves workflow turn to ConversationMemory.
        max_revisions         : Max review→writer cycles (default: settings.MAX_REVIEW_ITERATIONS).

        Returns
        -------
        Dict with keys: ok, request_id, session_id, workflow_status,
                        review, final_output, errors.
        """
        # Validate inputs
        errors = self._validate(user_input, content_type, objective)
        if errors:
            return self._failure(errors)

        content_type = content_type.lower()
        objective = objective.lower()
        session_id = session_id or str(uuid.uuid4())
        request_id = str(uuid.uuid4())

        # Optionally embed the brand hint so the Manager can resolve it
        resolved_input = f"[Brand: {brand}] {user_input}" if brand else user_input

        initial_state = self._build_state(
            request_id=request_id,
            session_id=session_id,
            user_input=resolved_input,
            content_type=content_type,
            platform="website",
            objective=objective,
            language=language,
            additional_instructions=additional_instructions,
            max_revisions=max_revisions or settings.MAX_REVIEW_ITERATIONS,
        )

        logger.info(
            "ContentWorkflow.run() | request_id=%s | content_type=%s | objective=%s",
            request_id, content_type, objective,
        )

        # ------------------------------------------------------------------
        # Execute the graph
        # ------------------------------------------------------------------
        try:
            final_state = self._graph.invoke(initial_state)
        except Exception as exc:
            logger.error("ContentWorkflow graph error: %s", exc, exc_info=True)
            return self._failure([f"Graph execution error: {exc}"])

        # ------------------------------------------------------------------
        # Save to ConversationMemory (non-fatal)
        # ------------------------------------------------------------------
        self._save_to_memory(session_id, user_input, final_state)

        return self._build_result(final_state)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(user_input: str, content_type: str, objective: str) -> List[str]:
        errors: List[str] = []
        if not user_input or not user_input.strip():
            errors.append("user_input cannot be empty.")
        if content_type.lower() not in _VALID_CONTENT_TYPES:
            errors.append(
                f"Invalid content_type '{content_type}'. "
                f"ContentWorkflow accepts: {sorted(_VALID_CONTENT_TYPES)}."
            )
        if objective.lower() not in _VALID_OBJECTIVES:
            errors.append(
                f"Invalid objective '{objective}'. "
                f"Valid values: {sorted(_VALID_OBJECTIVES)}."
            )
        return errors

    @staticmethod
    def _build_state(
        request_id: str,
        session_id: str,
        user_input: str,
        content_type: str,
        platform: str,
        objective: str,
        language: str,
        additional_instructions: str,
        max_revisions: int,
    ) -> Dict[str, Any]:
        return {
            # Identity
            "request_id": request_id,
            "session_id": session_id,
            # Request
            "user_input": user_input,
            "content_type": content_type,
            "platform": platform,
            "objective": objective,
            "language": language,
            "additional_instructions": additional_instructions,
            # Business context — populated by Manager Agent
            "brand_context": {},
            # Research — populated by Research Agent
            "research_data": {},
            "retrieved_documents": [],
            "sources": [],
            # Strategy — populated by Strategy Agent
            "strategy": {},
            "seo": {},
            "hashtags": [],
            # Writer — populated by Writer Agent
            "draft": "",
            "metadata": {},
            "formatted_output": {},
            "final_output": {},
            # Review — populated by Review Agent
            "review": {},
            # Workflow management
            "revision_count": 0,
            "max_revision_count": max_revisions,
            "current_agent": "",
            "next_agent": "manager",
            "workflow_status": "INIT",
            "errors": [],
        }

    @staticmethod
    def _save_to_memory(session_id: str, user_input: str, state: Dict[str, Any]) -> None:
        """Persist the workflow turn to ConversationMemory (swallows all errors)."""
        try:
            from memory.conversation_memory import ConversationMemory
            mem = ConversationMemory(session_id=session_id)
            mem.add_user_message(user_input)
            summary = (
                f"[ContentWorkflow] status={state.get('workflow_status')} | "
                f"score={state.get('review', {}).get('score', 'N/A')} | "
                f"words={state.get('metadata', {}).get('word_count', 'N/A')}"
            )
            mem.add_assistant_message(summary)
            mem.save_workflow_state(state)
        except Exception as exc:
            logger.warning("ConversationMemory save failed (non-fatal): %s", exc)

    @staticmethod
    def _build_result(state: Dict[str, Any]) -> Dict[str, Any]:
        review = state.get("review", {})
        return {
            "ok": state.get("workflow_status") == "COMPLETED",
            "request_id": state.get("request_id", ""),
            "session_id": state.get("session_id", ""),
            "workflow_status": state.get("workflow_status", "FAILED"),
            "review": {
                "score": review.get("score", 0),
                "status": review.get("status", ""),
                "needs_revision": review.get("needs_revision", False),
                "feedback": review.get("feedback", []),
                "issues": review.get("issues", []),
                "dimension_scores": review.get("dimension_scores", {}),
            },
            "revision_count": state.get("revision_count", 0),
            "metadata": state.get("metadata", {}),
            "final_output": state.get("final_output", {}),
            "errors": state.get("errors", []),
        }

    @staticmethod
    def _failure(errors: List[str]) -> Dict[str, Any]:
        logger.error("ContentWorkflow input validation failed: %s", errors)
        return {
            "ok": False,
            "request_id": "",
            "session_id": "",
            "workflow_status": "FAILED",
            "review": {},
            "revision_count": 0,
            "metadata": {},
            "final_output": {},
            "errors": errors,
        }
