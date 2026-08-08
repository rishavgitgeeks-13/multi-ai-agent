"""
Social Workflow
===============

Orchestrates the 5-agent pipeline for social media content:
  - LinkedIn posts  (content_type="linkedin", ~600 words)
  - Carousels       (content_type="carousel", ~800 words, multi-slide)
  - X (Twitter)     (content_type="linkedin" + platform="x", short-form)

Social content is short-form and engagement-driven. This workflow:
  1. Maps the caller's `platform` choice to the correct content_type.
  2. Injects platform-specific formatting instructions into writer prompts.
  3. Limits max revisions to 2 (social content converges quickly).
  4. Post-processes the draft to extract:
       - Hashtags (from final_output or trailing hashtag lines in the draft)
       - Character count (for X platform validation)
       - Engagement hook (first line / hook sentence of the post)
       - Slide count (carousel only)

Platforms supported
-------------------
  - "linkedin"   : Professional post (~600 words). H2 headings replaced
                   by spacing. Ends with 5–10 hashtags.
  - "carousel"   : LinkedIn/Instagram carousel (~800 words, 6–10 slides).
                   Format: **Slide N: Headline** + bullet points.
  - "x"          : X (Twitter) thread or single post. Short-form
                   (~280–2000 chars for threads). 2–3 hashtags.

Usage
-----
    from workflows.social_workflow import SocialWorkflow

    workflow = SocialWorkflow()

    # LinkedIn post
    result = workflow.run(
        user_input="Why AI agents are the next competitive advantage for SMBs",
        platform="linkedin",
        brand="Futuristix",
        session_id="sess-abc123",
    )

    # Carousel
    result = workflow.run(
        user_input="5 signs your operations need AI automation",
        platform="carousel",
        brand="Futuristix",
    )

    if result["ok"]:
        content    = result["final_output"]["content"]["markdown"]
        hashtags   = result["social_meta"]["hashtags"]
        hook       = result["social_meta"]["engagement_hook"]
"""

import logging
import re
import uuid
from typing import Any, Dict, List, Optional

from config.settings import settings
from graphs.graph import graph

logger = logging.getLogger(__name__)

# Platform → (content_type, word_count_hint, formatting_instructions)
_PLATFORM_CONFIG: Dict[str, Dict[str, Any]] = {
    "linkedin": {
        "content_type": "linkedin",
        "platform_label": "linkedin",
        "word_count": "500 to 700",
        "instructions": (
            "LINKEDIN FORMAT: "
            "Start with a powerful single sentence hook (no hashtags on the first line). "
            "Use short paragraphs (1 to 3 lines) separated by blank lines. "
            "No Markdown ## headings. LinkedIn renders plain text. "
            "End with 5 to 10 relevant hashtags on the last line. "
            "Tone: conversational but authoritative. "
            "Never use hyphen or dash characters anywhere in the post."
        ),
    },
    "carousel": {
        "content_type": "carousel",
        "platform_label": "linkedin",
        "word_count": "700 to 900",
        "instructions": (
            "CAROUSEL FORMAT: "
            "Format each slide as '**Slide N: Headline**' followed by 2 to 3 bullet points. "
            "Slide 1 = hook/title slide. Last slide = CTA slide. "
            "Each slide max 40 words. Strong visual language. Each slide must work standalone. "
            "Total slides: 6 to 10. Never use hyphen or dash characters."
        ),
    },
    "x": {
        "content_type": "linkedin",
        "platform_label": "x",
        "word_count": "200 to 350",
        "instructions": (
            "X (TWITTER) FORMAT: "
            "Write a thread of 4 to 8 tweets. Number each tweet: '1/', '2/', etc. "
            "Each tweet max 280 characters including the number prefix. "
            "First tweet is the hook and must stand alone as a single post. "
            "Last tweet is the CTA. Max 2 to 3 hashtags total (last tweet only). "
            "No Markdown headings. Never use hyphen or dash characters."
        ),
    },
    "instagram": {
        "content_type": "linkedin",
        "platform_label": "instagram",
        "word_count": "150 to 300",
        "instructions": (
            "INSTAGRAM CAPTION FORMAT: "
            "Start with a scroll stopping hook in the first line. "
            "Use short paragraphs and line breaks for mobile reading. "
            "Include a clear CTA near the end. "
            "End with 10 to 20 relevant hashtags on the last lines. "
            "No Markdown ## headings. Never use hyphen or dash characters."
        ),
    },
    "facebook": {
        "content_type": "linkedin",
        "platform_label": "facebook",
        "word_count": "200 to 400",
        "instructions": (
            "FACEBOOK POST FORMAT: "
            "Conversational hook first. Short paragraphs. "
            "Encourage comments with a question near the end. "
            "End with 3 to 8 relevant hashtags. "
            "No Markdown ## headings. Never use hyphen or dash characters."
        ),
    },
    "reddit": {
        "content_type": "linkedin",
        "platform_label": "reddit",
        "word_count": "250 to 500",
        "instructions": (
            "REDDIT POST FORMAT: "
            "Write like a helpful community member, not a brand ad. "
            "Lead with the useful insight or question. "
            "Use short sections and plain language. "
            "Soft brand mention only if relevant. "
            "Include 2 to 5 topical hashtags or community tags at the end if natural. "
            "No Markdown ## headings. Never use hyphen or dash characters."
        ),
    },
    "comment": {
        "content_type": "comment",
        "platform_label": "comment",
        "word_count": "25 to 80",
        "instructions": (
            "SOCIAL COMMENT FORMAT (reply under someone else's post): "
            "Write ONLY a natural comment/reply the user can paste under a post. "
            "INTENT LOCK: Match the user's intent exactly "
            "(agree, add a useful insight, ask a follow up, congratulate, gently correct, or soft brand mention). "
            "If the user pasted a post or described a post, reply to THAT post — do not invent a new post. "
            "2 to 5 short sentences. Conversational first person or direct address is fine. "
            "NO hashtags. NO SEO keywords list. NO titles/headings. NO bullet lists. "
            "NO CTA buttons, NO 'Book a call' hard sell, NO brand dump, NO hashtag block. "
            "Optional soft brand mention only if the user asked for it and it fits in one short phrase. "
            "Never use hyphen or dash characters."
        ),
    },
}

_VALID_PLATFORMS = set(_PLATFORM_CONFIG.keys())
_VALID_OBJECTIVES = {"engagement", "authority", "leads"}

# Regex to find trailing hashtag lines (one or more #tags)
_HASHTAG_LINE_RE = re.compile(r"((?:#\w+\s*){2,})", re.MULTILINE)
_SLIDE_RE = re.compile(r"\*\*Slide\s+\d+", re.IGNORECASE)


class SocialWorkflow:
    """
    Social media content workflow for LinkedIn, X, Instagram, Facebook,
    Reddit, comment replies, and carousels.

    Maps the caller's `platform` to the correct content_type and injects
    platform-specific formatting instructions into every writer call.
    """

    def __init__(self) -> None:
        self._graph = graph

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        user_input: str,
        platform: str = "linkedin",
        brand: Optional[str] = None,
        objective: str = "engagement",
        language: str = "English",
        additional_instructions: str = "",
        session_id: Optional[str] = None,
        max_revisions: int = 2,
    ) -> Dict[str, Any]:
        """
        Run the social media content workflow.

        Parameters
        ----------
        user_input            : Topic or brief for the post/carousel/thread.
        platform              : "linkedin" | "carousel" | "x" | "instagram" |
                                "facebook" | "reddit" | "comment".
        brand                 : Optional brand name or alias.
        objective             : "engagement" (default) | "authority" | "leads".
        language              : "English" (default) | "Hindi".
        additional_instructions: Extra writer guidance appended after platform rules.
        session_id            : If provided, saves workflow turn to ConversationMemory.
        max_revisions         : Review cycles (default 2 — social content is short).

        Returns
        -------
        Dict with keys: ok, request_id, session_id, workflow_status, review,
                        final_output, social_meta, errors.
        """
        errors = self._validate(user_input, platform, objective)
        if errors:
            return self._failure(errors)

        platform = platform.lower()
        objective = objective.lower()
        session_id = session_id or str(uuid.uuid4())
        request_id = str(uuid.uuid4())

        config = _PLATFORM_CONFIG[platform]
        content_type = config["content_type"]
        platform_label = config["platform_label"]

        # Build platform-aware writer instructions
        platform_instructions = config["instructions"]
        if platform == "comment":
            platform_instructions += (
                " Output must be paste ready as a comment only — nothing else."
            )
        if additional_instructions:
            # Keep length asks (e.g. "10 words") visible to Writer + Manager merge
            platform_instructions += " " + additional_instructions
            # Also prefix onto user_input so safety constraint extraction sees it
            # when Manager only scans user_input (belt-and-suspenders with Manager merge).
            if re.search(r"\b\d{1,5}\s*[\-]?\s*words?\b", additional_instructions, re.I):
                user_input = f"{user_input.strip()} ({additional_instructions.strip()})"

        # For comments, keep brand hint soft so Writer does not turn it into a promo post
        if platform == "comment":
            resolved_input = (
                f"[Brand: {brand}] [Format: social comment reply] {user_input}"
                if brand
                else f"[Format: social comment reply] {user_input}"
            )
        else:
            resolved_input = f"[Brand: {brand}] {user_input}" if brand else user_input

        initial_state = self._build_state(
            request_id=request_id,
            session_id=session_id,
            user_input=resolved_input,
            brand=brand,
            content_type=content_type,
            platform=platform_label,
            objective=objective,
            language=language,
            additional_instructions=platform_instructions,
            max_revisions=max_revisions,
        )

        logger.info(
            "SocialWorkflow.run() | request_id=%s | platform=%s | content_type=%s",
            request_id, platform, content_type,
        )

        # ------------------------------------------------------------------
        # Execute the graph
        # ------------------------------------------------------------------
        try:
            final_state = self._graph.invoke(initial_state)
        except Exception as exc:
            logger.error("SocialWorkflow graph error: %s", exc, exc_info=True)
            return self._failure([f"Graph execution error: {exc}"])

        # ------------------------------------------------------------------
        # Post-process: extract social-specific metadata
        # ------------------------------------------------------------------
        draft = final_state.get("draft", "") or ""
        hashtags = final_state.get("hashtags", []) or []
        if platform == "comment":
            # Comments never ship hashtags — strip if model still added any
            hashtags = []
            draft = self._strip_comment_noise(draft)
            final_state["draft"] = draft
            final_state["hashtags"] = []
            if isinstance(final_state.get("final_output"), dict):
                fo = dict(final_state["final_output"])
                fo["hashtags"] = []
                content = fo.get("content") or {}
                if isinstance(content, dict) and content.get("markdown"):
                    content = dict(content)
                    content["markdown"] = draft
                    fo["content"] = content
                final_state["final_output"] = fo

        social_meta = self._extract_social_meta(
            draft=draft,
            hashtags=hashtags,
            platform=platform,
        )

        # ------------------------------------------------------------------
        # Save to ConversationMemory (non-fatal)
        # ------------------------------------------------------------------
        self._save_to_memory(session_id, user_input, final_state, platform)

        return self._build_result(final_state, social_meta)

    # ------------------------------------------------------------------
    # Social-specific post-processing
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_comment_noise(draft: str) -> str:
        """Remove hashtags, headings, and CTA/hashtag footers from a comment draft."""
        if not draft:
            return draft
        lines: List[str] = []
        for line in draft.splitlines():
            stripped = line.strip()
            if not stripped:
                if lines and lines[-1] != "":
                    lines.append("")
                continue
            # Drop markdown headings / slide labels
            if re.match(r"^#{1,6}\s+", stripped):
                continue
            if stripped.lower().startswith("hashtags:"):
                continue
            # Drop pure hashtag lines
            if re.fullmatch(r"(?:#\w+\s*)+", stripped):
                continue
            # Remove inline trailing hashtag clusters
            cleaned = re.sub(r"(?:\s#\w+){1,}\s*$", "", stripped).strip()
            if cleaned:
                lines.append(cleaned)
        text = "\n".join(lines).strip()
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    @staticmethod
    def _extract_social_meta(
        draft: str,
        hashtags: List[str],
        platform: str,
    ) -> Dict[str, Any]:
        """
        Extract social media metadata from the draft and state.

        Returns
        -------
        Dict with:
          - engagement_hook   : first non-blank line of the draft
          - hashtags          : from state or extracted from draft trailing lines
          - character_count   : total draft length (critical for X)
          - slide_count       : number of slides (carousel only, else 0)
          - platform          : the platform that was used
        """
        if not draft:
            return {
                "engagement_hook": "",
                "hashtags": hashtags,
                "character_count": 0,
                "slide_count": 0,
                "platform": platform,
            }

        # Engagement hook: first non-blank, non-hashtag line
        hook = ""
        for line in draft.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                # Strip Markdown bold markers if present
                hook = re.sub(r"\*\*(.+?)\*\*", r"\1", stripped)
                hook = hook.strip("*_ ")
                break

        # Hashtags: prefer state (from HashtagService), fallback to draft extraction
        if not hashtags:
            all_tags: List[str] = []
            for match in _HASHTAG_LINE_RE.finditer(draft):
                tags = re.findall(r"#\w+", match.group(1))
                all_tags.extend(tags)
            hashtags = list(dict.fromkeys(all_tags))  # dedupe, preserve order

        # Character count (raw draft length, no Markdown rendering)
        character_count = len(draft)

        # Slide count (carousel only)
        slide_count = len(_SLIDE_RE.findall(draft)) if platform == "carousel" else 0

        return {
            "engagement_hook": hook,
            "hashtags": hashtags,
            "character_count": character_count,
            "slide_count": slide_count,
            "platform": platform,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(user_input: str, platform: str, objective: str) -> List[str]:
        errors: List[str] = []
        if not user_input or not user_input.strip():
            errors.append("user_input cannot be empty.")
        if platform.lower() not in _VALID_PLATFORMS:
            errors.append(
                f"Invalid platform '{platform}'. "
                f"SocialWorkflow accepts: {sorted(_VALID_PLATFORMS)}."
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
        brand: Optional[str],
        content_type: str,
        platform: str,
        objective: str,
        language: str,
        additional_instructions: str,
        max_revisions: int,
    ) -> Dict[str, Any]:
        return {
            "request_id": request_id,
            "session_id": session_id,
            "user_input": user_input,
            "content_type": content_type,
            "brand": brand,
            "platform": platform,
            "objective": objective,
            "language": language,
            "additional_instructions": additional_instructions,
            "primary_topic": "",
            "user_constraints": {},
            "safety": {},
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
    def _save_to_memory(
        session_id: str,
        user_input: str,
        state: Dict[str, Any],
        platform: str,
    ) -> None:
        try:
            from memory.conversation_memory import ConversationMemory
            mem = ConversationMemory(session_id=session_id)
            mem.add_user_message(
                user_input,
                metadata={"workflow": "social", "platform": platform},
            )
            draft = ""
            final = state.get("final_output") or {}
            content = final.get("content") or {}
            if isinstance(content, dict):
                draft = str(content.get("markdown") or "")
            hashtags = final.get("hashtags") or state.get("hashtags") or []
            summary = (
                f"[SocialWorkflow:{platform}] status={state.get('workflow_status')} | "
                f"score={state.get('review', {}).get('score', 'N/A')} | "
                f"hashtags={len(hashtags)}\n\n"
                f"{draft[:2500]}"
            )
            if hashtags:
                summary += "\n\nHashtags: " + " ".join(str(h) for h in hashtags)
            mem.add_assistant_message(
                summary,
                metadata={
                    "workflow": "social",
                    "platform": platform,
                    "hashtags": hashtags,
                },
            )
            mem.save_workflow_state(state)
        except Exception as exc:
            logger.warning("ConversationMemory save failed (non-fatal): %s", exc)

    @staticmethod
    def _build_result(state: Dict[str, Any], social_meta: Dict[str, Any]) -> Dict[str, Any]:
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
            "social_meta": social_meta,
            "errors": state.get("errors", []),
            "safety": state.get("safety") or {},
            "primary_topic": state.get("primary_topic") or "",
            "user_constraints": state.get("user_constraints") or {},
        }

    @staticmethod
    def _failure(errors: List[str]) -> Dict[str, Any]:
        logger.error("SocialWorkflow input validation failed: %s", errors)
        return {
            "ok": False,
            "request_id": "",
            "session_id": "",
            "workflow_status": "FAILED",
            "review": {},
            "revision_count": 0,
            "metadata": {},
            "final_output": {},
            "social_meta": {},
            "errors": errors,
        }
