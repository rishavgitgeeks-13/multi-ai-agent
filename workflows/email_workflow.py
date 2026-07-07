"""
Email Workflow
==============

Orchestrates the 5-agent pipeline for email content generation.

Email campaigns are short-form (~400 words), persuasion-focused, and have
a different success profile than long-form SEO content:
  - Objective  : leads (default) | engagement
  - Platform   : email
  - Content type: email
  - Word count : 300–600 words
  - Max revisions: 2 (shorter content needs less iteration)

Campaign types supported
------------------------
  - newsletter     : value-delivery, curated insight, brand authority
  - nurture        : relationship-building, problem-aware content, soft CTA
  - promotional    : offer-led, urgency-driven, hard CTA
  - transactional  : event-triggered, factual, clear next-step CTA

The workflow adds email-specific post-processing on top of the core pipeline:
  - Extracts subject line from the draft (first non-blank line after "Subject:")
  - Derives preview text (first 140 characters of the email body)
  - Flags personalization tokens like [First Name], [Company]

Usage
-----
    from workflows.email_workflow import EmailWorkflow

    workflow = EmailWorkflow()

    result = workflow.run(
        user_input="Announce our new AI audit service to founders",
        brand="Futuristix",
        campaign_type="promotional",
        session_id="sess-abc123",
    )

    if result["ok"]:
        body    = result["final_output"]["content"]["markdown"]
        subject = result["email_meta"]["subject_line"]
        preview = result["email_meta"]["preview_text"]
"""

import logging
import re
import uuid
from typing import Any, Dict, List, Optional

from config.settings import settings
from graphs.graph import graph

logger = logging.getLogger(__name__)

# Valid campaign types — injected into additional_instructions so the Writer
# understands the email's purpose without extra state fields.
_CAMPAIGN_TYPES = {"newsletter", "nurture", "promotional", "transactional"}
_VALID_OBJECTIVES = {"leads", "engagement"}

# Token patterns that indicate personalization placeholders
_PERSONALIZATION_RE = re.compile(r"\[([A-Z][a-zA-Z\s]+)\]")


class EmailWorkflow:
    """
    Email content workflow: short-form, persuasion-driven email generation.

    Sets `content_type="email"`, `platform="email"`, `objective="leads"` by
    default and limits review cycles to 2 (emails are short enough that
    2 passes produce publication-ready output in practice).
    """

    def __init__(self) -> None:
        self._graph = graph

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        user_input: str,
        brand: Optional[str] = None,
        campaign_type: str = "newsletter",
        objective: str = "leads",
        language: str = "English",
        additional_instructions: str = "",
        session_id: Optional[str] = None,
        max_revisions: int = 2,
    ) -> Dict[str, Any]:
        """
        Run the email content workflow.

        Parameters
        ----------
        user_input            : Topic, offer, or brief for the email.
        brand                 : Optional brand name or alias.
        campaign_type         : "newsletter" | "nurture" | "promotional" | "transactional".
        objective             : "leads" (default) | "engagement".
        language              : "English" (default) | "Hindi".
        additional_instructions: Extra writer guidance appended to the prompt.
        session_id            : If provided, saves workflow turn to ConversationMemory.
        max_revisions         : Review cycles before forcing PASS (default 2).

        Returns
        -------
        Dict with keys: ok, request_id, session_id, workflow_status, review,
                        final_output, email_meta, errors.
        """
        errors = self._validate(user_input, campaign_type, objective)
        if errors:
            return self._failure(errors)

        campaign_type = campaign_type.lower()
        objective = objective.lower()
        session_id = session_id or str(uuid.uuid4())
        request_id = str(uuid.uuid4())

        # Embed campaign type into the writer instructions so the LLM knows
        # what kind of email to produce without a new state field.
        campaign_hint = (
            f"This is a {campaign_type.upper()} email campaign. "
            f"Tone and CTA should reflect the campaign type. "
        )
        full_instructions = campaign_hint + additional_instructions

        resolved_input = f"[Brand: {brand}] {user_input}" if brand else user_input

        initial_state = self._build_state(
            request_id=request_id,
            session_id=session_id,
            user_input=resolved_input,
            objective=objective,
            language=language,
            additional_instructions=full_instructions,
            max_revisions=max_revisions,
        )

        logger.info(
            "EmailWorkflow.run() | request_id=%s | campaign_type=%s | objective=%s",
            request_id, campaign_type, objective,
        )

        # ------------------------------------------------------------------
        # Execute the graph
        # ------------------------------------------------------------------
        try:
            final_state = self._graph.invoke(initial_state)
        except Exception as exc:
            logger.error("EmailWorkflow graph error: %s", exc, exc_info=True)
            return self._failure([f"Graph execution error: {exc}"])

        # ------------------------------------------------------------------
        # Post-process: extract email-specific metadata from the draft
        # ------------------------------------------------------------------
        email_meta = self._extract_email_meta(
            draft=final_state.get("draft", ""),
            campaign_type=campaign_type,
        )

        # ------------------------------------------------------------------
        # Save to ConversationMemory (non-fatal)
        # ------------------------------------------------------------------
        self._save_to_memory(session_id, user_input, final_state)

        return self._build_result(final_state, email_meta)

    # ------------------------------------------------------------------
    # Email-specific post-processing
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_email_meta(draft: str, campaign_type: str) -> Dict[str, Any]:
        """
        Extract email metadata from the Markdown draft produced by WriterService.

        WriterService formats emails as:
          Subject: <subject line>
          Hi [First Name],
          <body>

        This method extracts:
          - subject_line : first "Subject: …" line, or first non-blank line
          - preview_text : first 140 characters of the body
          - personalization_tokens : list of [Token] placeholders
          - campaign_type : passed through
        """
        if not draft:
            return {
                "subject_line": "",
                "preview_text": "",
                "personalization_tokens": [],
                "campaign_type": campaign_type,
            }

        lines = draft.strip().splitlines()
        subject_line = ""
        body_start = 0

        # Look for explicit "Subject: ..." line
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.lower().startswith("subject:"):
                subject_line = stripped[len("subject:"):].strip()
                body_start = i + 1
                break

        # Fallback: use first non-blank line as subject
        if not subject_line:
            for i, line in enumerate(lines):
                if line.strip():
                    subject_line = line.strip().lstrip("#").strip()
                    body_start = i + 1
                    break

        # Preview text: first 140 chars of body (skip greeting line)
        body_lines = [l for l in lines[body_start:] if l.strip()]
        body_text = " ".join(body_lines)
        # Skip salutation-style openers ("Hi [First Name]," / "Dear …,")
        salutation_re = re.compile(r"^(hi|hello|dear|greetings)[,\s]", re.IGNORECASE)
        body_for_preview = body_text
        for line in body_lines:
            if not salutation_re.match(line.strip()):
                body_for_preview = " ".join(body_lines[body_lines.index(line):])
                break
        preview_text = re.sub(r"\s+", " ", body_for_preview)[:140].strip()

        # Personalization tokens
        tokens = list(dict.fromkeys(_PERSONALIZATION_RE.findall(draft)))

        return {
            "subject_line": subject_line,
            "preview_text": preview_text,
            "personalization_tokens": tokens,
            "campaign_type": campaign_type,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(user_input: str, campaign_type: str, objective: str) -> List[str]:
        errors: List[str] = []
        if not user_input or not user_input.strip():
            errors.append("user_input cannot be empty.")
        if campaign_type.lower() not in _CAMPAIGN_TYPES:
            errors.append(
                f"Invalid campaign_type '{campaign_type}'. "
                f"Valid values: {sorted(_CAMPAIGN_TYPES)}."
            )
        if objective.lower() not in _VALID_OBJECTIVES:
            errors.append(
                f"Invalid objective '{objective}'. "
                f"EmailWorkflow accepts: {sorted(_VALID_OBJECTIVES)}."
            )
        return errors

    @staticmethod
    def _build_state(
        request_id: str,
        session_id: str,
        user_input: str,
        objective: str,
        language: str,
        additional_instructions: str,
        max_revisions: int,
    ) -> Dict[str, Any]:
        return {
            "request_id": request_id,
            "session_id": session_id,
            "user_input": user_input,
            "content_type": "email",
            "platform": "email",
            "objective": objective,
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
            summary = (
                f"[EmailWorkflow] status={state.get('workflow_status')} | "
                f"score={state.get('review', {}).get('score', 'N/A')}"
            )
            mem.add_assistant_message(summary)
            mem.save_workflow_state(state)
        except Exception as exc:
            logger.warning("ConversationMemory save failed (non-fatal): %s", exc)

    @staticmethod
    def _build_result(state: Dict[str, Any], email_meta: Dict[str, Any]) -> Dict[str, Any]:
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
            "email_meta": email_meta,
            "errors": state.get("errors", []),
        }

    @staticmethod
    def _failure(errors: List[str]) -> Dict[str, Any]:
        logger.error("EmailWorkflow input validation failed: %s", errors)
        return {
            "ok": False,
            "request_id": "",
            "session_id": "",
            "workflow_status": "FAILED",
            "review": {},
            "revision_count": 0,
            "metadata": {},
            "final_output": {},
            "email_meta": {},
            "errors": errors,
        }
