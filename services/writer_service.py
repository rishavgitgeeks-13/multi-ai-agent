"""
Writer Service
==============

Generates the content draft for the Writer Agent.

Input:
    user_input    : str
    research_data : Dict  — documents, sources, statistics, citations
    strategy      : Dict  — title, content_angle, outline, keywords,
                            tone, audience, pain_points, cta
    brand_context : Dict  — display_name, tone, reader_segment,
                            pain_points, keyword_direction, cta

Output:
    draft : str — full content in Markdown

Pipeline:
    1. Resolve content type (blog | article | linkedin | email | carousel)
    2. Resolve or generate the content outline
    3. Extract usable research context (stats, citations)
    4a. Long-form (blog, article)  → write section-by-section, then assemble
    4b. Short-form (linkedin, email, carousel) → write in one shot
    5. Return the Markdown draft string

This service does NOT perform research, SEO scoring, or strategy planning.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from anthropic import Anthropic

from config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Content-type configuration
# ---------------------------------------------------------------------------

SHORT_FORM_TYPES = {"linkedin", "email", "carousel"}
LONG_FORM_TYPES = {"blog", "article"}

WORD_COUNT_TARGETS: Dict[str, int] = {
    "blog": 1800,
    "article": 2200,
    "linkedin": 600,
    "email": 400,
    "carousel": 800,
}

# Research stats injected per section prompt to ground the LLM
_MAX_STATS_PER_SECTION = 3
_MAX_CITATIONS_GLOBAL = 5
# Words from the previous section tail passed for narrative continuity
_CONTINUITY_TAIL_WORDS = 150


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ContentSection:
    """One section of the content outline."""

    heading: str
    heading_level: int  # 2 = H2, 3 = H3
    brief: str          # what to cover in this section
    keywords: List[str] = field(default_factory=list)


@dataclass
class ContentOutline:
    """Complete plan for the content piece, consumed by every write method."""

    title: str
    content_angle: str
    audience: str
    tone: str
    cta: str
    sections: List[ContentSection]


# ---------------------------------------------------------------------------
# WriterService
# ---------------------------------------------------------------------------


class WriterService:
    """Produces the full-length Markdown draft from the strategy package."""

    def __init__(self) -> None:
        self._anthropic = Anthropic()
        self._model = settings.ANTHROPIC_MODEL
        self._temperature = settings.DEFAULT_TEMPERATURE
        self._max_tokens = settings.MAX_TOKENS
        logger.info("WriterService ready | model=%s", self._model)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        user_input: str,
        research_data: Dict,
        strategy: Dict,
        brand_context: Dict,
    ) -> str:
        """Generate and return the full Markdown content draft."""
        logger.info("WriterService.run() | query=%s…", user_input[:80])

        content_type = self._resolve_content_type(strategy, brand_context)
        platform = (
            strategy.get("platform")
            or brand_context.get("platform")
            or "website"
        )

        outline = self._resolve_outline(
            strategy=strategy,
            brand_context=brand_context,
            user_input=user_input,
            content_type=content_type,
        )

        research_ctx = self._build_research_context(research_data)
        rewrite_instruction = str(strategy.get("rewrite_instruction", "")).strip()

        if content_type in SHORT_FORM_TYPES:
            draft = self._write_short_form(
                outline=outline,
                research_ctx=research_ctx,
                content_type=content_type,
                platform=platform,
                rewrite_instruction=rewrite_instruction,
            )
        else:
            draft = self._write_long_form(
                outline=outline,
                research_ctx=research_ctx,
                content_type=content_type,
                rewrite_instruction=rewrite_instruction,
            )

        logger.info(
            "WriterService complete | content_type=%s | words=%d",
            content_type,
            len(draft.split()),
        )
        return draft

    # ------------------------------------------------------------------
    # Step 1 — Content type resolution
    # ------------------------------------------------------------------

    def _resolve_content_type(
        self,
        strategy: Dict,
        brand_context: Dict,
    ) -> str:
        """Return the normalised content type, defaulting to 'article'."""
        ct = (
            strategy.get("content_type")
            or brand_context.get("content_type")
            or "article"
        )
        return str(ct).lower().strip()

    # ------------------------------------------------------------------
    # Step 2 — Outline resolution
    # ------------------------------------------------------------------

    def _resolve_outline(
        self,
        strategy: Dict,
        brand_context: Dict,
        user_input: str,
        content_type: str,
    ) -> ContentOutline:
        """Use the strategy outline when present; otherwise generate one via LLM."""
        existing = strategy.get("outline", [])

        if existing and isinstance(existing, list) and len(existing) > 0:
            logger.info("Using outline from strategy (%d sections)", len(existing))
            return self._parse_strategy_outline(strategy, brand_context)

        logger.info("Strategy outline empty — generating via LLM")
        return self._generate_outline_via_llm(
            user_input=user_input,
            strategy=strategy,
            brand_context=brand_context,
            content_type=content_type,
        )

    def _parse_strategy_outline(
        self,
        strategy: Dict,
        brand_context: Dict,
    ) -> ContentOutline:
        """Convert the raw strategy outline list into a typed ContentOutline."""
        sections: List[ContentSection] = []

        for item in strategy.get("outline", []):
            if isinstance(item, dict):
                sections.append(ContentSection(
                    heading=str(item.get("heading") or item.get("title") or "Section"),
                    heading_level=int(item.get("heading_level") or item.get("level") or 2),
                    brief=str(item.get("brief") or item.get("description") or ""),
                    keywords=list(item.get("keywords", [])),
                ))
            elif isinstance(item, str) and item.strip():
                sections.append(ContentSection(
                    heading=item.strip(),
                    heading_level=2,
                    brief="",
                ))

        audience = strategy.get("audience") or brand_context.get("reader_segment", [])
        audience_str = (
            ", ".join(str(a) for a in audience)
            if isinstance(audience, list)
            else str(audience)
        )

        return ContentOutline(
            title=strategy.get("title", ""),
            content_angle=strategy.get("content_angle", ""),
            audience=audience_str,
            tone=strategy.get("tone") or brand_context.get("tone") or "professional",
            cta=strategy.get("cta") or brand_context.get("cta") or "",
            sections=sections,
        )

    def _generate_outline_via_llm(
        self,
        user_input: str,
        strategy: Dict,
        brand_context: Dict,
        content_type: str,
    ) -> ContentOutline:
        """Generate a full ContentOutline using the Anthropic model."""
        target_words = WORD_COUNT_TARGETS.get(content_type, 1800)
        n_sections = "2–4" if content_type in SHORT_FORM_TYPES else "4–7"

        audience = strategy.get("audience") or brand_context.get("reader_segment", [])
        audience_str = (
            ", ".join(str(a) for a in audience)
            if isinstance(audience, list)
            else str(audience)
        )
        tone = strategy.get("tone") or brand_context.get("tone") or "professional"
        keywords = strategy.get("keywords") or brand_context.get("keyword_direction", [])
        pain_points = strategy.get("pain_points") or brand_context.get("pain_points", [])

        prompt = f"""Create a detailed content outline for a {content_type}.

USER QUERY      : {user_input}
BRAND           : {brand_context.get("display_name") or brand_context.get("brand", "")}
CONTENT ANGLE   : {strategy.get("content_angle", "")}
TONE            : {tone}
AUDIENCE        : {audience_str}
KEY KEYWORDS    : {", ".join(str(k) for k in keywords[:10]) or "none"}
PAIN POINTS     : {"; ".join(str(p) for p in pain_points[:5]) or "none"}
CTA             : {strategy.get("cta") or brand_context.get("cta", "")}
TARGET WORDS    : ~{target_words}

Return a JSON object with this exact schema:
{{
  "title": "<compelling H1 title containing the primary keyword>",
  "content_angle": "<unique hook or angle for this piece>",
  "sections": [
    {{
      "heading": "<section heading>",
      "heading_level": 2,
      "brief": "<1–2 sentences: what this section must cover>",
      "keywords": ["<kw1>", "<kw2>"]
    }}
  ]
}}

Rules:
- {n_sections} sections
- Sections flow: problem → solution → proof → CTA
- Headings are benefit-driven and keyword-rich
- Each brief is specific enough to write a full section from
- Return ONLY the JSON object — no prose, no markdown fences
"""
        try:
            raw = self._call_llm(
                system=(
                    "You are an expert content strategist. "
                    "Create precise, structured content outlines. "
                    "Return valid JSON only — no prose, no markdown."
                ),
                user=prompt,
            )
            return self._parse_outline_json(raw, strategy, brand_context)
        except Exception as exc:
            logger.error("Outline LLM call failed: %s — using fallback outline", exc)
            return self._fallback_outline(user_input, strategy, brand_context)

    def _parse_outline_json(
        self,
        raw: str,
        strategy: Dict,
        brand_context: Dict,
    ) -> ContentOutline:
        """Parse LLM JSON response into a typed ContentOutline."""
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.error("Outline JSON parse error: %s | raw_head=%s", exc, cleaned[:300])
            raise

        sections = [
            ContentSection(
                heading=str(s.get("heading", "Section")),
                heading_level=int(s.get("heading_level", 2)),
                brief=str(s.get("brief", "")),
                keywords=list(s.get("keywords", [])),
            )
            for s in data.get("sections", [])
            if isinstance(s, dict)
        ]

        audience = strategy.get("audience") or brand_context.get("reader_segment", [])
        audience_str = (
            ", ".join(str(a) for a in audience)
            if isinstance(audience, list)
            else str(audience)
        )

        return ContentOutline(
            title=str(data.get("title", "")),
            content_angle=str(data.get("content_angle", "")),
            audience=audience_str,
            tone=strategy.get("tone") or brand_context.get("tone") or "professional",
            cta=strategy.get("cta") or brand_context.get("cta") or "",
            sections=sections,
        )

    def _fallback_outline(
        self,
        user_input: str,
        strategy: Dict,
        brand_context: Dict,
    ) -> ContentOutline:
        """Rule-based fallback when LLM outline generation fails."""
        audience = strategy.get("audience") or brand_context.get("reader_segment", [])
        audience_str = (
            ", ".join(str(a) for a in audience)
            if isinstance(audience, list)
            else str(audience)
        )
        return ContentOutline(
            title=user_input[:80],
            content_angle="Practical guide",
            audience=audience_str,
            tone=strategy.get("tone") or brand_context.get("tone") or "professional",
            cta=strategy.get("cta") or brand_context.get("cta") or "",
            sections=[
                ContentSection("The Core Challenge", 2, "Define the problem and why it matters.", []),
                ContentSection("Why Existing Approaches Fall Short", 2, "Gaps in current solutions.", []),
                ContentSection("The Solution", 2, "Practical approach with concrete steps.", []),
                ContentSection("Key Benefits & Outcomes", 2, "Measurable results readers can expect.", []),
                ContentSection("Getting Started", 2, "Actionable first steps for the reader.", []),
            ],
        )

    # ------------------------------------------------------------------
    # Step 3 — Research context extraction
    # ------------------------------------------------------------------

    def _build_research_context(self, research_data: Dict) -> Dict:
        """Pull statistics and citations from the research package."""
        stats = [
            str(s).strip()
            for s in research_data.get("statistics", [])
            if str(s).strip()
        ]
        citations = [
            str(c).strip()
            for c in research_data.get("citations", [])
            if str(c).strip()
        ]
        return {
            "stats": stats,
            "citations": citations[:_MAX_CITATIONS_GLOBAL],
        }

    def _pick_stats(self, research_ctx: Dict, n: int = 3) -> str:
        """Format up to n stats for injection into a section prompt."""
        selected = research_ctx.get("stats", [])[:n]
        if not selected:
            return "No specific stats available — use credible domain knowledge."
        return "\n".join(f"- {s}" for s in selected)

    # ------------------------------------------------------------------
    # Step 4a — Long-form writing (blog, article)
    # ------------------------------------------------------------------

    def _write_long_form(
        self,
        outline: ContentOutline,
        research_ctx: Dict,
        content_type: str,
        rewrite_instruction: str = "",
    ) -> str:
        """Write introduction, body sections, and conclusion; then assemble."""
        introduction = self._write_introduction(
            outline=outline,
            research_ctx=research_ctx,
            rewrite_instruction=rewrite_instruction,
        )

        section_bodies: List[str] = []
        previous_tail = self._tail(introduction, _CONTINUITY_TAIL_WORDS)

        for section in outline.sections:
            body = self._write_section(
                section=section,
                outline=outline,
                research_ctx=research_ctx,
                previous_tail=previous_tail,
                rewrite_instruction=rewrite_instruction,
            )
            section_bodies.append(body)
            previous_tail = self._tail(body, _CONTINUITY_TAIL_WORDS)

        conclusion = self._write_conclusion(outline, content_type, rewrite_instruction)

        return self._assemble_long_form(
            outline=outline,
            introduction=introduction,
            section_bodies=section_bodies,
            conclusion=conclusion,
        )

    def _write_introduction(
        self,
        outline: ContentOutline,
        research_ctx: Dict,
        rewrite_instruction: str = "",
    ) -> str:
        """Write a hook-driven introduction (no heading, flows after the H1)."""
        prompt = f"""Write the introduction for a content piece.

TITLE           : {outline.title}
CONTENT ANGLE   : {outline.content_angle}
AUDIENCE        : {outline.audience}
TONE            : {outline.tone}

SECTIONS AHEAD:
{self._format_section_list(outline.sections)}

RELEVANT STATS:
{self._pick_stats(research_ctx, n=2)}

Requirements:
- 100–150 words
- Open with a powerful hook: a bold claim, surprising stat, or sharp question
- State the core problem the reader faces
- Promise the value this piece delivers
- Do NOT include a heading — this flows directly after the H1 title
- Tone: {outline.tone}
- No meta-commentary ("In this article we will…")
- Plain Markdown only

Write the introduction:
"""
        return self._call_llm(
            system=self._system_prompt(outline, rewrite_instruction),
            user=prompt,
        )

    def _write_section(
        self,
        section: ContentSection,
        outline: ContentOutline,
        research_ctx: Dict,
        previous_tail: str,
        rewrite_instruction: str = "",
    ) -> str:
        """Write one body section with full narrative context."""
        kw_str = ", ".join(section.keywords) if section.keywords else "none specified"

        prompt = f"""Write one body section of a {outline.tone} content piece.

ARTICLE TITLE   : {outline.title}
AUDIENCE        : {outline.audience}
TONE            : {outline.tone}

THIS SECTION:
  Heading (H{section.heading_level}) : {section.heading}
  Must cover     : {section.brief}
  Keywords       : {kw_str}

PREVIOUS SECTION ENDED WITH:
"{previous_tail}"

RESEARCH STATS TO DRAW FROM:
{self._pick_stats(research_ctx, n=_MAX_STATS_PER_SECTION)}

Requirements:
- Start with {'##' if section.heading_level == 2 else '###'} {section.heading}
- 200–350 words
- Add H3 subheadings if the section covers multiple distinct points
- Naturally include 1–2 of the target keywords
- Use bullet points or numbered lists where they improve clarity
- Include at least one concrete example, stat, or data point if available
- End with a sentence that transitions naturally toward the next topic
- No filler openers ("In this section…", "Now let's look at…")
- Tone: {outline.tone}

Write this section:
"""
        return self._call_llm(
            system=self._system_prompt(outline, rewrite_instruction),
            user=prompt,
        )

    def _write_conclusion(
        self,
        outline: ContentOutline,
        content_type: str,
        rewrite_instruction: str = "",
    ) -> str:
        """Write a conclusion that synthesises the piece and closes with a CTA."""
        prompt = f"""Write the conclusion for a {content_type}.

TITLE           : {outline.title}
CONTENT ANGLE   : {outline.content_angle}
CTA             : {outline.cta}
TONE            : {outline.tone}

SECTIONS COVERED:
{self._format_section_list(outline.sections)}

Requirements:
- Start with ## Conclusion (Markdown H2)
- 100–150 words
- Recap the core insight in 1–2 sentences — no new information
- Tell the reader exactly what to do next
- Close with a clear, action-oriented CTA: {outline.cta}
- Tone: {outline.tone}

Write the conclusion:
"""
        return self._call_llm(
            system=self._system_prompt(outline, rewrite_instruction),
            user=prompt,
        )

    def _assemble_long_form(
        self,
        outline: ContentOutline,
        introduction: str,
        section_bodies: List[str],
        conclusion: str,
    ) -> str:
        """Join all pieces into a single coherent Markdown document."""
        parts = [f"# {outline.title}", "", introduction.strip(), ""]
        for body in section_bodies:
            parts.append(body.strip())
            parts.append("")
        parts.append(conclusion.strip())
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Step 4b — Short-form writing (linkedin, email, carousel)
    # ------------------------------------------------------------------

    def _write_short_form(
        self,
        outline: ContentOutline,
        research_ctx: Dict,
        content_type: str,
        platform: str,
        rewrite_instruction: str = "",
    ) -> str:
        """Write the entire short-form piece in a single LLM call."""
        target_words = WORD_COUNT_TARGETS.get(content_type, 600)
        format_rules = self._format_rules(content_type, platform)

        prompt = f"""Write a complete {content_type} for {platform}.

TITLE / TOPIC   : {outline.title}
CONTENT ANGLE   : {outline.content_angle}
AUDIENCE        : {outline.audience}
TONE            : {outline.tone}
CTA             : {outline.cta}
TARGET LENGTH   : ~{target_words} words

CONTENT STRUCTURE TO COVER:
{self._format_section_list(outline.sections)}

RELEVANT STATS:
{self._pick_stats(research_ctx, n=3)}

FORMAT REQUIREMENTS:
{format_rules}

Write the complete {content_type}:
"""
        return self._call_llm(
            system=self._system_prompt(outline, rewrite_instruction),
            user=prompt,
        )

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    def _system_prompt(
        self,
        outline: ContentOutline,
        rewrite_instruction: str = "",
    ) -> str:
        """Shared system prompt that frames the LLM as a focused writer."""
        base = (
            f"You are an expert content writer specialising in {outline.tone.lower()} writing "
            f"for {outline.audience}. "
            "You follow formatting instructions exactly, never add meta-commentary, "
            "and return only the requested content — no preamble, no sign-off."
        )
        if rewrite_instruction:
            base += (
                f"\n\nREVISION INSTRUCTIONS FROM EDITOR:\n{rewrite_instruction}\n"
                "Apply these instructions throughout the entire piece."
            )
        return base

    def _format_section_list(self, sections: List[ContentSection]) -> str:
        """Format the section list for inclusion in a prompt."""
        return "\n".join(
            f"  {i + 1}. {s.heading}" + (f" — {s.brief}" if s.brief else "")
            for i, s in enumerate(sections)
        )

    def _format_rules(self, content_type: str, platform: str) -> str:
        """Return platform-specific formatting instructions."""
        if content_type == "linkedin":
            return (
                "- First line: single bold hook (no hashtags)\n"
                "- Short paragraphs (1–3 lines) separated by blank lines\n"
                "- No markdown headers (##) — LinkedIn renders plain text\n"
                "- End with 3–5 relevant hashtags on their own line"
            )
        if content_type == "email":
            return (
                "- First line: Subject: <compelling subject line>\n"
                "- Greeting: Hi [First Name],\n"
                "- Body paragraphs 2–4 sentences max\n"
                "- One CTA: [CTA TEXT](URL)\n"
                "- Sign-off: Best, [Sender Name]"
            )
        if content_type == "carousel":
            return (
                "- Format each slide as **Slide N: <Headline>**\n"
                "- Each slide: 1 headline + 2–3 bullet points\n"
                "- Slide 1 = hook/title slide\n"
                "- Last slide = CTA slide\n"
                "- Each slide ≤ 40 words"
            )
        return (
            "- Markdown headings (##, ###)\n"
            "- Paragraphs 3–5 sentences\n"
            "- Bullet/numbered lists for multi-item points\n"
            "- Bold key terms on first use"
        )

    # ------------------------------------------------------------------
    # Anthropic wrapper
    # ------------------------------------------------------------------

    def _call_llm(
        self,
        system: str,
        user: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Invoke the Anthropic model and return the plain text response."""
        response = self._anthropic.messages.create(
            model=self._model,
            max_tokens=max_tokens or self._max_tokens,
            temperature=self._temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text

    # ------------------------------------------------------------------
    # String utility
    # ------------------------------------------------------------------

    @staticmethod
    def _tail(text: str, n_words: int) -> str:
        """Return the last n_words of text for narrative continuity context."""
        words = text.split()
        return " ".join(words[-n_words:]) if len(words) > n_words else text
