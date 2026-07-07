"""
SEO Workflow
============

Orchestrates the 5-agent pipeline with an explicit SEO objective, producing
long-form content (article or blog) alongside a detailed SEO analysis report.

Difference from ContentWorkflow
--------------------------------
ContentWorkflow is general-purpose; SEOWorkflow is tuned for organic search:

  1. Objective is always "seo".
  2. Returns an enriched `seo_analysis` block alongside the content, including:
       - Full keyword scores (all 6 dimensions per keyword)
       - Keyword density computed over the published draft
       - Technical SEO checklist (title length, meta length, slug format)
       - Internal SEO audit of the draft (heading structure, keyword placement)
  3. The additional_instructions inject explicit SEO guidance into every writer
     call (primary keyword in H1, keyword density targets, heading hierarchy).

Usage
-----
    from workflows.seo_workflow import SEOWorkflow

    workflow = SEOWorkflow()

    result = workflow.run(
        user_input="AI agents for small business automation",
        content_type="article",
        brand="Futuristix",
        session_id="sess-abc123",
    )

    if result["ok"]:
        content  = result["final_output"]["content"]["markdown"]
        analysis = result["seo_analysis"]
        print(f"Primary keyword: {analysis['primary_keyword']}")
        print(f"Keyword density: {analysis['keyword_density']}")
        print(f"Technical SEO:   {analysis['technical_checklist']}")
"""

import logging
import re
import uuid
from typing import Any, Dict, List, Optional

from config.settings import settings
from graphs.graph import graph

logger = logging.getLogger(__name__)

_VALID_CONTENT_TYPES = {"article", "blog"}

# SEO writer guidance appended to every SEO workflow run
_SEO_WRITER_INSTRUCTIONS = (
    "SEO REQUIREMENTS: "
    "Include the primary keyword in the H1 title, at least one H2 heading, "
    "and distribute it naturally across 3–4 body paragraphs. "
    "Target keyword density: 1–2% for primary, 0.5–1% for secondary keywords. "
    "Use H2 for major sections and H3 for sub-points — no skipped heading levels. "
    "Ensure the conclusion contains a standalone CTA sentence."
)


class SEOWorkflow:
    """
    SEO-focused long-form content workflow.

    Returns the standard content result plus `seo_analysis` — a detailed
    breakdown of keyword scores, density metrics, and a technical SEO checklist
    derived from the published draft and SEO blueprint.
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
        language: str = "English",
        additional_instructions: str = "",
        session_id: Optional[str] = None,
        max_revisions: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run the SEO content workflow.

        Parameters
        ----------
        user_input            : Topic, search query, or content brief.
        content_type          : "article" (default, ~2200 words) | "blog" (~1800 words).
        brand                 : Optional brand name or alias.
        language              : "English" (default) | "Hindi".
        additional_instructions: Extra writer guidance (appended after SEO instructions).
        session_id            : If provided, saves workflow turn to ConversationMemory.
        max_revisions         : Review cycles (default: settings.MAX_REVIEW_ITERATIONS).

        Returns
        -------
        Dict with keys: ok, request_id, session_id, workflow_status, review,
                        final_output, seo_analysis, errors.
        """
        errors = self._validate(user_input, content_type)
        if errors:
            return self._failure(errors)

        content_type = content_type.lower()
        session_id = session_id or str(uuid.uuid4())
        request_id = str(uuid.uuid4())

        # Merge SEO-specific writer instructions with caller's custom instructions
        merged_instructions = _SEO_WRITER_INSTRUCTIONS
        if additional_instructions:
            merged_instructions += " " + additional_instructions

        resolved_input = f"[Brand: {brand}] {user_input}" if brand else user_input

        initial_state = self._build_state(
            request_id=request_id,
            session_id=session_id,
            user_input=resolved_input,
            content_type=content_type,
            language=language,
            additional_instructions=merged_instructions,
            max_revisions=max_revisions or settings.MAX_REVIEW_ITERATIONS,
        )

        logger.info(
            "SEOWorkflow.run() | request_id=%s | content_type=%s",
            request_id, content_type,
        )

        # ------------------------------------------------------------------
        # Execute the graph
        # ------------------------------------------------------------------
        try:
            final_state = self._graph.invoke(initial_state)
        except Exception as exc:
            logger.error("SEOWorkflow graph error: %s", exc, exc_info=True)
            return self._failure([f"Graph execution error: {exc}"])

        # ------------------------------------------------------------------
        # Build the SEO analysis report from the completed state
        # ------------------------------------------------------------------
        seo_analysis = self._build_seo_analysis(final_state)

        # ------------------------------------------------------------------
        # Save to ConversationMemory (non-fatal)
        # ------------------------------------------------------------------
        self._save_to_memory(session_id, user_input, final_state)

        return self._build_result(final_state, seo_analysis)

    # ------------------------------------------------------------------
    # SEO analysis builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_seo_analysis(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a detailed SEO analysis report from the final state.

        Combines the SEO blueprint (from StrategyAgent) with draft-level
        metrics (keyword density, heading audit) and a technical checklist.
        """
        seo_blueprint = state.get("seo", {})
        draft = state.get("draft", "")
        metadata = state.get("metadata", {})

        primary_keywords: List[str] = seo_blueprint.get("primary_keywords", [])
        secondary_keywords: List[str] = seo_blueprint.get("secondary_keywords", [])
        primary_keyword = primary_keywords[0] if primary_keywords else ""

        # ------------------------------------------------------------------
        # Keyword density (occurrences / total_words)
        # ------------------------------------------------------------------
        total_words = metadata.get("word_count") or (len(draft.split()) if draft else 0)
        density: Dict[str, float] = {}
        if draft and total_words:
            draft_lower = draft.lower()
            for kw in primary_keywords + secondary_keywords:
                count = len(re.findall(re.escape(kw.lower()), draft_lower))
                density[kw] = round(count / total_words * 100, 2)

        # ------------------------------------------------------------------
        # Heading audit
        # ------------------------------------------------------------------
        h1_matches = re.findall(r"^# .+", draft, re.MULTILINE)
        h2_matches = re.findall(r"^## .+", draft, re.MULTILINE)
        h3_matches = re.findall(r"^### .+", draft, re.MULTILINE)

        keyword_in_h1 = any(
            primary_keyword.lower() in h.lower() for h in h1_matches
        ) if primary_keyword else False
        keyword_in_h2 = any(
            primary_keyword.lower() in h.lower() for h in h2_matches
        ) if primary_keyword else False

        heading_audit = {
            "h1_count": len(h1_matches),
            "h2_count": len(h2_matches),
            "h3_count": len(h3_matches),
            "primary_keyword_in_h1": keyword_in_h1,
            "primary_keyword_in_h2": keyword_in_h2,
        }

        # ------------------------------------------------------------------
        # Technical SEO checklist
        # ------------------------------------------------------------------
        meta_title = seo_blueprint.get("meta_title", "")
        meta_description = seo_blueprint.get("meta_description", "")
        slug = seo_blueprint.get("slug", "")

        technical_checklist = {
            "meta_title_present": bool(meta_title),
            "meta_title_length_ok": 10 <= len(meta_title) <= 60 if meta_title else False,
            "meta_description_present": bool(meta_description),
            "meta_description_length_ok": 50 <= len(meta_description) <= 160 if meta_description else False,
            "slug_present": bool(slug),
            "slug_format_ok": bool(re.match(r"^[a-z0-9-]+$", slug)) if slug else False,
            "h1_present": len(h1_matches) == 1,
            "h2_minimum_met": len(h2_matches) >= 2,
            "word_count_in_range": (
                1200 <= total_words <= 2500 if total_words else False
            ),
            "primary_keyword_in_meta_title": (
                primary_keyword.lower() in meta_title.lower()
                if primary_keyword and meta_title else False
            ),
        }

        # Overall SEO score: fraction of passed checks (0–100)
        checks_passed = sum(1 for v in technical_checklist.values() if v is True)
        total_checks = len(technical_checklist)
        seo_score = round(checks_passed / total_checks * 100) if total_checks else 0

        return {
            "primary_keyword": primary_keyword,
            "primary_keywords": primary_keywords,
            "secondary_keywords": secondary_keywords,
            "search_intent": seo_blueprint.get("search_intent", ""),
            "meta_title": meta_title,
            "meta_description": meta_description,
            "slug": slug,
            "keyword_scores": seo_blueprint.get("keyword_scores", []),
            "keyword_density": density,
            "heading_audit": heading_audit,
            "technical_checklist": technical_checklist,
            "seo_score": seo_score,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(user_input: str, content_type: str) -> List[str]:
        errors: List[str] = []
        if not user_input or not user_input.strip():
            errors.append("user_input cannot be empty.")
        if content_type.lower() not in _VALID_CONTENT_TYPES:
            errors.append(
                f"Invalid content_type '{content_type}'. "
                f"SEOWorkflow accepts: {sorted(_VALID_CONTENT_TYPES)}."
            )
        return errors

    @staticmethod
    def _build_state(
        request_id: str,
        session_id: str,
        user_input: str,
        content_type: str,
        language: str,
        additional_instructions: str,
        max_revisions: int,
    ) -> Dict[str, Any]:
        return {
            "request_id": request_id,
            "session_id": session_id,
            "user_input": user_input,
            "content_type": content_type,
            "platform": "website",
            "objective": "seo",
            "language": language,
            "additional_instructions": additional_instructions,
            "brand_context": {},
            "research_data": {},
            "retrieved_documents": [],
            "sources": [],
            "strategy": {},
            "seo": {},
            "hashtags": [],
            "draft": "",
            "metadata": {},
            "formatted_output": {},
            "final_output": {},
            "review": {},
            "revision_count": 0,
            "max_revision_count": max_revisions,
            "current_agent": "",
            "next_agent": "manager",
            "workflow_status": "INIT",
            "errors": [],
        }

    @staticmethod
    def _save_to_memory(session_id: str, user_input: str, state: Dict[str, Any]) -> None:
        try:
            from memory.conversation_memory import ConversationMemory
            mem = ConversationMemory(session_id=session_id)
            mem.add_user_message(user_input)
            seo = state.get("seo", {})
            summary = (
                f"[SEOWorkflow] status={state.get('workflow_status')} | "
                f"score={state.get('review', {}).get('score', 'N/A')} | "
                f"primary_keyword={seo.get('primary_keywords', [''])[0] if seo.get('primary_keywords') else 'N/A'}"
            )
            mem.add_assistant_message(summary)
            mem.save_workflow_state(state)
        except Exception as exc:
            logger.warning("ConversationMemory save failed (non-fatal): %s", exc)

    @staticmethod
    def _build_result(state: Dict[str, Any], seo_analysis: Dict[str, Any]) -> Dict[str, Any]:
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
            "seo_analysis": seo_analysis,
            "errors": state.get("errors", []),
        }

    @staticmethod
    def _failure(errors: List[str]) -> Dict[str, Any]:
        logger.error("SEOWorkflow input validation failed: %s", errors)
        return {
            "ok": False,
            "request_id": "",
            "session_id": "",
            "workflow_status": "FAILED",
            "review": {},
            "revision_count": 0,
            "metadata": {},
            "final_output": {},
            "seo_analysis": {},
            "errors": errors,
        }
