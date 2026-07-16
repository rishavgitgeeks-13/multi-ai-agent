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
    4a. Long-form (blog, article)  → one-shot full draft
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
_CONTINUITY_TAIL_WORDS = 50


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
        # Fail fast if Claude credentials are missing.
        if not settings.ANTHROPIC_API_KEY:
            raise ValueError(
                "ANTHROPIC_API_KEY is not configured."
            )

        # Authenticated Anthropic client.
        self._anthropic = Anthropic(
            api_key=settings.ANTHROPIC_API_KEY
        )

        self._model = settings.ANTHROPIC_MODEL
        self._temperature = settings.DEFAULT_TEMPERATURE
        self._max_tokens = settings.MAX_TOKENS

        logger.info(
            "WriterService ready | model=%s",
            self._model,
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        user_input: str,
        research_data: Dict,
        strategy: Dict,
        brand_context: Dict,
        previous_draft: str = "",
        primary_topic: str = "",
        additional_instructions: str = "",
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
        primary_keywords, secondary_keywords = self._resolve_seo_keywords(strategy)
        target_words = self._resolve_target_words(content_type, strategy)
        topic_lock = (primary_topic or strategy.get("primary_topic") or user_input or "").strip()

        # On revision: surgically edit the existing draft instead of regenerating.
        if rewrite_instruction and previous_draft.strip() and content_type in LONG_FORM_TYPES:
            draft = self._revise_long_form(
                previous_draft=previous_draft,
                outline=outline,
                research_ctx=research_ctx,
                content_type=content_type,
                rewrite_instruction=rewrite_instruction,
                primary_keywords=primary_keywords,
                secondary_keywords=secondary_keywords,
            )
        elif content_type in SHORT_FORM_TYPES:
            draft = self._write_short_form(
                outline=outline,
                research_ctx=research_ctx,
                content_type=content_type,
                platform=platform,
                rewrite_instruction=rewrite_instruction,
                primary_keywords=primary_keywords,
                secondary_keywords=secondary_keywords,
                target_words=target_words,
                primary_topic=topic_lock,
                additional_instructions=additional_instructions,
            )
        else:
            draft = self._write_long_form(
                outline=outline,
                research_ctx=research_ctx,
                content_type=content_type,
                rewrite_instruction=rewrite_instruction,
                primary_keywords=primary_keywords,
                secondary_keywords=secondary_keywords,
                target_words=target_words,
                primary_topic=topic_lock,
                additional_instructions=additional_instructions,
            )

        # Deterministic quality boost: ensure attributed research stats are present.
        if content_type in LONG_FORM_TYPES and target_words >= 400:
            draft = self._enrich_factual_grounding(
                draft=draft,
                research_ctx=research_ctx,
                outline=outline,
                secondary_keywords=secondary_keywords,
            )

        logger.info(
            "WriterService complete | content_type=%s | words=%d | target=%s",
            content_type,
            len(draft.split()),
            target_words,
        )

        logger.info(
            "Draft generated:\n%s",
            draft[:3000]
        )

        return draft

    @staticmethod
    def _resolve_target_words(content_type: str, strategy: Dict) -> int:
        """Prefer user-requested word count; else content-type default."""
        user_target = strategy.get("target_word_count")
        if user_target is not None:
            try:
                n = int(user_target)
                if 1 <= n <= 50000:
                    return n
            except (TypeError, ValueError):
                pass
        return WORD_COUNT_TARGETS.get(content_type, 1800)

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
        return self._generate_outline(
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

    def _generate_outline(
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
        primary_keywords = (
            strategy.get("keywords")
            or brand_context.get("keyword_direction", [])
        )
        secondary_keywords = strategy.get("secondary_keywords") or []
        pain_points = strategy.get("pain_points") or brand_context.get("pain_points", [])

        primary_str = ", ".join(str(k) for k in primary_keywords[:3]) or "none"
        secondary_str = ", ".join(str(k) for k in secondary_keywords[:6]) or "none"

        prompt = f"""Create a detailed content outline for a {content_type}.

USER QUERY      : {user_input}
BRAND           : {brand_context.get("display_name") or brand_context.get("brand", "")}
CONTENT ANGLE   : {strategy.get("content_angle", "")}
TONE            : {tone}
AUDIENCE        : {audience_str}
PRIMARY KEYWORDS: {primary_str}
SECONDARY KEYWORDS: {secondary_str}
PAIN POINTS     : {"; ".join(str(p) for p in pain_points[:5]) or "none"}
CTA             : {strategy.get("cta") or brand_context.get("cta", "")}
TARGET WORDS    : ~{target_words}

Return a JSON object with this exact schema:
{{
  "title": "<compelling H1 title containing the first primary keyword>",
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
- H1 title MUST include the first primary keyword naturally
- At least one H2 heading should include a primary or secondary keyword
- Assign each section 1 primary-or-secondary keyword in "keywords" (do not invent new ones)
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
            return (
                "No statistics are available.\n"
                "Do NOT invent percentages, benchmarks, revenue figures, "
                "survey data, or numerical claims.\n"
                "Do NOT make absolute industry claims without a citation from "
                "CITATIONS AVAILABLE; hedge or omit instead."
            )
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
        primary_keywords: Optional[List[str]] = None,
        secondary_keywords: Optional[List[str]] = None,
        target_words: Optional[int] = None,
        primary_topic: str = "",
        additional_instructions: str = "",
    ) -> str:
        """Write the full long-form piece in a single LLM call (token-efficient)."""
        if target_words is None:
            target_words = WORD_COUNT_TARGETS.get(content_type, 1800)
        primary = [str(k) for k in (primary_keywords or []) if str(k).strip()][:2]
        secondary = [str(k) for k in (secondary_keywords or []) if str(k).strip()][:6]
        primary_str = ", ".join(primary) or "none"
        secondary_str = ", ".join(secondary) or "none"
        lead_primary = primary[0] if primary else ""

        citations_block = (
            "\n".join(f"- {c}" for c in research_ctx.get("citations", [])[:8])
            or "none"
        )
        stats_n = 8 if rewrite_instruction else 8
        revision_block = ""
        if rewrite_instruction:
            revision_block = f"""
CRITICAL REVISION PASS — you must apply these editor notes:
{rewrite_instruction}

Mandatory fixes for this revision (do not skip):
- Embed at least 3 statistics from RESEARCH STATS below, each with clear attribution
  including source name AND a concrete figure/year when present in the snippet
  (e.g. "According to <Source> (Year): …XX%…").
- Never invent organisation names, report titles, years, or percentages.
- If a research snippet is vague, either quote it exactly with attribution or omit it.
- Remove absolute uncited claims ("most startups fail…") unless they appear in stats/citations.
- Place at least 1 secondary keyword in the introduction and 1 in the conclusion.
- End with the exact CTA phrase: {outline.cta}
- Every sentence must be complete — no mid-sentence cutoffs.
- Write currency as "USD 500" / "USD 1,000" — never use the $ character.
"""

        topic_block = ""
        if primary_topic:
            topic_block = f"""
PRIMARY TOPIC LOCK (mandatory — do not change meaning, roles, or subject):
{primary_topic}
"""

        extra_block = ""
        if additional_instructions.strip():
            extra_block = f"\nADDITIONAL USER INSTRUCTIONS:\n{additional_instructions.strip()}\n"

        # Micro / short long-form when user asks for very few words
        length_rules = (
            f"TARGET LENGTH   : ~{target_words} words — hit this length closely"
            if target_words < 400
            else f"TARGET LENGTH   : ~{target_words} words"
        )

        prompt = f"""Write a complete {content_type} in Markdown.
{revision_block}{topic_block}{extra_block}
TITLE           : {outline.title}
CONTENT ANGLE   : {outline.content_angle}
AUDIENCE        : {outline.audience}
TONE            : {outline.tone}
CTA (use verbatim): {outline.cta}
{length_rules}
PRIMARY KEYWORDS: {primary_str}
SECONDARY KEYWORDS: {secondary_str}

OUTLINE TO FOLLOW:
{self._format_section_list(outline.sections)}

RESEARCH STATS (use only these — do not invent figures):
{self._pick_stats(research_ctx, n=stats_n)}

CITATIONS AVAILABLE:
{citations_block}

SEO placement rules (mandatory):
- Start with `# {outline.title}` — H1 must include "{lead_primary or 'the primary keyword'}"
- Use the first primary keyword in the introduction (first 100 words)
- Use each primary keyword at least once in body copy
- Use at least 2 secondary keywords naturally across H2s or body (no stuffing)
- Place at least 1 secondary keyword naturally in the introduction AND 1 in the conclusion
- At least one `##` heading should contain a primary or secondary keyword

Content rules:
- Write like a human, not an AI: vary sentence and paragraph length, use natural transitions and contractions, avoid clichéd filler phrases (no "in today's fast-paced world", "moreover", "furthermore", "in conclusion", "it's worth noting", "dive in", "game-changer", "a testament to", "unlock the power")
- Stay strictly on the PRIMARY TOPIC LOCK — never invert victims/roles or change the subject
- Write a hook-driven introduction (100–150 words, no heading under the H1) unless target length is under 400 words — then keep intro proportional
- Cover every outline section as `##` headings (scale section length to hit ~{target_words} words total)
- Complete every sentence — never stop mid-word or mid-sentence
- End with `## Conclusion` that recaps and closes with the exact CTA: {outline.cta}
- Prefer CTA wording like "Book an AI Discovery Call" style specificity — do not use vague "reach out today"
- When RESEARCH STATS lists any items and target length >= 400, embed at least 3 attributed statistics in the article body
  (intro or early body, one mid-article, one in proof/closing). Format: "According to <Source>: <figure>…"
- When a proof / case-study / real-world section appears in the outline, ground it with research stats or named citations above — do not use brand name alone as proof
- Never invent percentages, benchmarks, financial figures, organisation names, or report titles
- Do not state absolute industry claims (e.g. "most startups fail because…") unless that exact claim appears in RESEARCH STATS or CITATIONS AVAILABLE; otherwise hedge or omit
- Write money amounts as "USD 500" or "USD 50,000" — never use the $ character (breaks Markdown renderers)
- Match brand tone exactly throughout: {outline.tone}
- Return ONLY Markdown — no preamble

Write the complete {content_type}:
"""
        # 8192 avoids mid-article truncation for ~1800–2200 word pieces
        max_tok = 2048 if target_words < 400 else 8192
        return self._call_llm(
            system=self._system_prompt(outline, rewrite_instruction, primary_topic),
            user=prompt,
            max_tokens=max_tok,
        )

    def _revise_long_form(
        self,
        previous_draft: str,
        outline: ContentOutline,
        research_ctx: Dict,
        content_type: str,
        rewrite_instruction: str,
        primary_keywords: Optional[List[str]] = None,
        secondary_keywords: Optional[List[str]] = None,
    ) -> str:
        """Edit an existing draft against review feedback (preserve structure)."""
        primary = [str(k) for k in (primary_keywords or []) if str(k).strip()][:2]
        secondary = [str(k) for k in (secondary_keywords or []) if str(k).strip()][:6]
        primary_str = ", ".join(primary) or "none"
        secondary_str = ", ".join(secondary) or "none"
        citations_block = (
            "\n".join(f"- {c}" for c in research_ctx.get("citations", [])[:8])
            or "none"
        )
        # Keep revision prompt within context: prefer full draft when possible.
        draft_for_edit = previous_draft
        if len(draft_for_edit) > 14000:
            draft_for_edit = previous_draft[:7000] + "\n\n…\n\n" + previous_draft[-5000:]

        prompt = f"""Revise the existing {content_type} Markdown. Do NOT rewrite from scratch.

EDITOR FEEDBACK (must fix):
{rewrite_instruction}

Preserve the overall structure, headings, and voice. Make targeted edits only.

BRAND TONE (exact) : {outline.tone}
AUDIENCE           : {outline.audience}
CTA (verbatim)     : {outline.cta}
PRIMARY KEYWORDS   : {primary_str}
SECONDARY KEYWORDS : {secondary_str}

RESEARCH STATS (use only these — do not invent figures):
{self._pick_stats(research_ctx, n=8)}

CITATIONS AVAILABLE:
{citations_block}

Mandatory edit checklist:
1. Insert at least 3 attributed statistics from RESEARCH STATS (named source + concrete figure).
   Prefer placing one in the intro, one in a mid-body/proof section, and one near the conclusion.
2. Remove or hedge absolute uncited claims; never invent org names/years/%.
3. Place at least 1 secondary keyword naturally in the introduction AND 1 in the conclusion body
   (not only in H2 headings).
4. Closing CTA must use verbatim: {outline.cta}
5. Match tone exactly: {outline.tone}
6. Write currency as "USD 500" — never use the $ character.
7. Complete every sentence.
8. Keep strong existing sections; only edit what the editor feedback requires.

EXISTING DRAFT:
{draft_for_edit}

Return the FULL revised Markdown article only — no preamble.
"""
        return self._call_llm(
            system=self._system_prompt(outline, rewrite_instruction),
            user=prompt,
            max_tokens=8192,
        )

    def _enrich_factual_grounding(
        self,
        draft: str,
        research_ctx: Dict,
        outline: ContentOutline,
        secondary_keywords: Optional[List[str]] = None,
    ) -> str:
        """
        Lightweight second pass: inject attributed research stats and
        secondary-keyword coverage without regenerating the article.
        """
        stats = [str(s).strip() for s in research_ctx.get("stats", []) if str(s).strip()]
        if not draft.strip() or not stats:
            return draft

        attribution_hits = len(
            re.findall(
                r"According to | \(\d{4}\)|Source:|HubSpot|Gartner|CB Insights|Salesforce|McKinsey",
                draft,
                re.IGNORECASE,
            )
        )
        secondary = [str(k) for k in (secondary_keywords or []) if str(k).strip()][:4]
        missing_secondary = [
            kw for kw in secondary
            if kw.lower() not in draft[:800].lower()
            or kw.lower() not in draft[-900:].lower()
        ]

        # Skip enrichment when draft already looks well grounded and complete.
        if attribution_hits >= 4 and not missing_secondary:
            return draft

        secondary_line = ", ".join(missing_secondary) if missing_secondary else "none"
        prompt = f"""Improve factual grounding of this Markdown article with MINIMAL edits.

Tone to preserve: {outline.tone}
CTA to preserve verbatim if present: {outline.cta}

RESEARCH STATS (only use these — do not invent):
{self._pick_stats(research_ctx, n=8)}

CITATIONS:
{chr(10).join(f"- {c}" for c in research_ctx.get("citations", [])[:6]) or "none"}

Required edits:
1. Ensure at least 3 clearly attributed statistics appear (intro, mid-body or proof, near close).
   Format: "According to <Source> (Year): <figure>…"
2. If listed, weave these secondary keywords naturally into intro and/or conclusion: {secondary_line}
3. Do not invent figures, organisations, or years.
4. Do not use the $ character — write USD amounts.
5. Keep structure/headings; return the FULL revised Markdown only.

DRAFT:
{draft[:12000]}
"""
        try:
            enriched = self._call_llm(
                system=(
                    "You are a careful editorial reviser. Make minimal targeted edits. "
                    "Return only the full Markdown article."
                ),
                user=prompt,
                max_tokens=8192,
            )
            return enriched.strip() or draft
        except Exception as exc:
            logger.warning("Factual grounding enrichment failed (non-fatal): %s", exc)
            return draft

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
- 150-200 words
- Add H3 subheadings if the section covers multiple distinct points
- Naturally include 1–2 of the target keywords
- Use bullet points or numbered lists where they improve clarity
- Include a statistic ONLY if it exists in the research context.
- Never invent percentages, benchmarks, or financial figures.
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
        primary_keywords: Optional[List[str]] = None,
        secondary_keywords: Optional[List[str]] = None,
        target_words: Optional[int] = None,
        primary_topic: str = "",
        additional_instructions: str = "",
    ) -> str:
        """Write the entire short-form piece in a single LLM call."""
        if target_words is None:
            target_words = WORD_COUNT_TARGETS.get(content_type, 600)
        format_rules = self._format_rules(content_type, platform)
        primary = [str(k) for k in (primary_keywords or []) if str(k).strip()][:2]
        secondary = [str(k) for k in (secondary_keywords or []) if str(k).strip()][:4]

        topic_block = ""
        if primary_topic:
            topic_block = (
                f"\nPRIMARY TOPIC LOCK (do not change meaning/roles):\n{primary_topic}\n"
            )
        extra_block = ""
        if additional_instructions.strip():
            extra_block = f"\nADDITIONAL USER INSTRUCTIONS:\n{additional_instructions.strip()}\n"
        rewrite_block = ""
        if rewrite_instruction.strip():
            rewrite_block = f"\nREVISION NOTES:\n{rewrite_instruction.strip()}\n"

        prompt = f"""Write a complete {content_type} for {platform}.
{topic_block}{extra_block}{rewrite_block}
TITLE / TOPIC   : {outline.title}
CONTENT ANGLE   : {outline.content_angle}
AUDIENCE        : {outline.audience}
TONE            : {outline.tone}
CTA             : {outline.cta}
TARGET LENGTH   : ~{target_words} words — adhere closely to this length
PRIMARY KEYWORDS: {", ".join(primary) or "none"}
SECONDARY KEYWORDS: {", ".join(secondary) or "none"}

CONTENT STRUCTURE TO COVER:
{self._format_section_list(outline.sections)}

RELEVANT STATS:
{self._pick_stats(research_ctx, n=3)}

FORMAT REQUIREMENTS:
{format_rules}

SEO notes:
- Include the first primary keyword early and naturally
- Weave 1–2 secondary keywords only if they fit the platform tone
- Do not keyword-stuff
- Stay strictly on the primary topic; never divert or invert roles

Human voice (important):
- Sound like a real person, not AI. Vary sentence length, use contractions, be specific
- Avoid clichés: no "in today's fast-paced world", "moreover", "furthermore", "in conclusion", "dive in", "game-changer", "unlock the power"

Write the complete {content_type}:
"""
        max_tok = 512 if target_words <= 50 else 2048
        return self._call_llm(
            system=self._system_prompt(outline, rewrite_instruction, primary_topic),
            user=prompt,
            max_tokens=max_tok,
        )

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_seo_keywords(strategy: Dict) -> tuple:
        """Return (primary, secondary) lists from strategy / nested seo blueprint."""
        seo = strategy.get("seo") or {}
        primary = (
            strategy.get("keywords")
            or seo.get("primary_keywords")
            or []
        )
        secondary = (
            strategy.get("secondary_keywords")
            or seo.get("secondary_keywords")
            or []
        )
        primary = [str(k).strip() for k in primary if str(k).strip()]
        secondary = [str(k).strip() for k in secondary if str(k).strip()]
        return primary, secondary

    def _system_prompt(
        self,
        outline: ContentOutline,
        rewrite_instruction: str = "",
        primary_topic: str = "",
    ) -> str:
        """Shared system prompt that frames the LLM as a focused, human-sounding writer."""
        base = (
            f"You are an experienced human content writer specialising in {outline.tone.lower()} writing "
            f"for {outline.audience}. "
            "You follow formatting instructions exactly, never add meta-commentary, "
            "and return only the requested content — no preamble, no sign-off. "
            "Never invent abusive, discriminatory, or illegal how-to content. "
            "Never divert from the user's primary topic or invert roles/meaning.\n\n"
            + self._human_voice_guide()
        )
        if primary_topic:
            base += f"\n\nPRIMARY TOPIC LOCK:\n{primary_topic}"
        if rewrite_instruction:
            base += (
                f"\n\nREVISION INSTRUCTIONS FROM EDITOR:\n{rewrite_instruction}\n"
                "Apply these instructions throughout the entire piece."
            )
        return base

    @staticmethod
    def _human_voice_guide() -> str:
        """Balanced, brand-safe rules that make output read as human-written."""
        return (
            "WRITE LIKE A HUMAN (critical — content must not read as AI-generated):\n"
            "- Vary sentence length and rhythm. Mix short, punchy sentences with longer ones. "
            "Avoid a uniform, robotic cadence.\n"
            "- Vary paragraph length too — some one-liners, some fuller paragraphs.\n"
            "- Use natural transitions. NEVER use these AI-cliché phrases: "
            "\"in today's fast-paced world\", \"in today's digital age\", \"in the ever-evolving\", "
            "\"when it comes to\", \"it's worth noting\", \"it's important to note\", \"needless to say\", "
            "\"moreover\", \"furthermore\", \"in conclusion\", \"in summary\", \"to sum up\", "
            "\"dive in\"/\"dive deep\", \"unlock the power\", \"unleash\", \"a game-changer\", "
            "\"a testament to\", \"plays a crucial/vital/pivotal role\", \"navigating the\", "
            "\"elevate your\", \"rest assured\", \"look no further\", \"we've got you covered\".\n"
            "- Use contractions naturally (it's, you're, don't, we've).\n"
            "- Prefer concrete, specific nouns and real examples over vague generalities.\n"
            "- Address the reader directly with \"you\" where it fits; light first-person (\"we\") is fine.\n"
            "- Do not over-hedge or over-explain. Trust the reader.\n"
            "- Avoid formulaic scaffolding (e.g. rigidly equal sections, a forced summary that "
            "restates everything). End with a genuine, specific closing rather than a generic wrap-up.\n"
            "- Keep it professional and on-brand — natural, not slangy or unprofessional.\n"
            "- No emojis unless explicitly requested."
        )

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
