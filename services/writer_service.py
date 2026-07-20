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

from openai import OpenAI
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
    brand_name: str = ""
    awareness_first: bool = False


# ---------------------------------------------------------------------------
# WriterService
# ---------------------------------------------------------------------------


class WriterService:
    """Produces the full-length Markdown draft from the strategy package."""

    def __init__(self) -> None:
        # Fail fast if OpenAI credentials are missing.
        if not settings.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not configured."
            )

        # Authenticated OpenAI client.
        self._openai = OpenAI(
            api_key=settings.OPENAI_API_KEY
        )

        self._model = settings.OPENAI_MODEL
        self._temperature = settings.DEFAULT_TEMPERATURE
        self._max_tokens = settings.MAX_TOKENS

        logger.info(
            "WriterService ready | model=%s",
            self._model,
        )

    @staticmethod
    def _brand_display_name(brand_context: Dict) -> str:
        return str(
            brand_context.get("display_name")
            or brand_context.get("brand")
            or ""
        ).strip()

    @staticmethod
    def _is_awareness_first(brand_context: Dict) -> bool:
        """
        Kinvo (and brands marked content_style=awareness_first) should educate
        before pitching. Other brands keep existing sales-friendly flow.
        """
        style = str(brand_context.get("content_style") or "").strip().lower()
        if style == "awareness_first":
            return True
        ns = str(
            brand_context.get("namespace") or brand_context.get("brand") or ""
        ).strip().lower()
        name = WriterService._brand_display_name(brand_context).lower()
        return ns == "kinvo" or "kinvo" in name

    @staticmethod
    def _awareness_first_rules(brand_name: str, cta: str) -> str:
        brand = (brand_name or "the brand").strip()
        cta_line = (cta or "").strip() or "the brand CTA"
        return f"""
AWARENESS-FIRST PACING (mandatory — write an awareness piece, not a sales brochure):
- Lead with the reader's real challenge, emotions, and practical guidance.
- Keep emotional connection strong: do not jump from the problem straight into product features.
- Do NOT mention {brand} in the introduction or in the first half of the body sections.
- Most of the article must remain useful education a reader can act on without buying.
- Introduce {brand} only in a late body section (near the end), as one concrete example of a structured approach — not the whole article.
- Place the CTA "{cta_line}" only in the conclusion (verbatim), not as a hard sell in every section.
- Avoid brochure language early (e.g. feature lists, "premium families choose us") until the late brand section.
"""

    @staticmethod
    def _stat_context_rules() -> str:
        return (
            "- When citing RESEARCH STATS, include source name plus year and/or scope "
            "(geography, sample, report name) when those details appear in the snippet. "
            "Do not invent missing year/scope; if absent, attribute what is available and avoid overclaiming."
        )

    # Meta / workflow phrases that must never be pasted into published copy.
    _LEAKY_KEYWORD_RE = re.compile(
        r"("
        r"\b(linkedin|twitter|x|instagram|facebook|carousel)\s+announc\w*\b|"
        r"\b(linkedin|carousel|email|x)\s+format\b|"
        r"\b(seo\s+requirements|additional\s+user\s+instructions|revision\s+notes)\b|"
        r"\bthis\s+is\s+a\s+(newsletter|nurture|promotional|transactional)\s+email\b|"
        r"\bcampaign\s+type\b|"
        r"\b(target\s+keyword\s+density|primary\s+keyword\s+in\s+the\s+h1)\b"
        r")",
        re.I,
    )

    @classmethod
    def _is_leaky_keyword(cls, keyword: str) -> bool:
        """True when a 'keyword' looks like prompt/meta text, not a searchable phrase."""
        kw = (keyword or "").strip()
        if not kw:
            return True
        words = kw.split()
        # Long run-ons are usually prompt fragments, not placeable SEO terms.
        if len(words) > 8:
            return True
        if cls._LEAKY_KEYWORD_RE.search(kw):
            return True
        # Platform label as the start of a multi-word "keyword" (e.g. "linkedin announcing…")
        if len(words) >= 2 and re.match(
            r"^(linkedin|twitter|instagram|facebook|carousel|newsletter)\b",
            kw,
            re.I,
        ):
            return True
        return False

    @classmethod
    def _filter_placeable_keywords(cls, keywords: Optional[List[str]], limit: int) -> List[str]:
        cleaned: List[str] = []
        for raw in keywords or []:
            kw = str(raw).strip()
            if not kw or cls._is_leaky_keyword(kw):
                continue
            if kw.lower() in {c.lower() for c in cleaned}:
                continue
            cleaned.append(kw)
            if len(cleaned) >= limit:
                break
        return cleaned

    @classmethod
    def _format_editorial_intent(cls, additional_instructions: str) -> str:
        """
        Present extra guidance as intent only so the model does not paste it
        into the draft (root cause of 'linkedin announcing…' style leaks).
        """
        text = (additional_instructions or "").strip()
        if not text:
            return ""
        # Cap size so huge instruction dumps are less likely to be echoed.
        if len(text) > 1200:
            text = text[:1200].rstrip() + "…"
        return (
            "\nEDITORIAL INTENT (follow the meaning only — "
            "NEVER copy, quote, or paraphrase this block into the published draft; "
            "never mention platform/format labels like 'LinkedIn announcing', "
            "'LINKEDIN FORMAT', 'SEO REQUIREMENTS', or campaign-type boilerplate):\n"
            f"{text}\n"
        )

    @staticmethod
    def _no_prompt_leak_rules() -> str:
        return (
            "- NEVER paste workflow meta into the draft: platform names as announcements "
            "('linkedin announcing…'), format labels, 'ADDITIONAL/EDITORIAL INTENT' text, "
            "SEO requirement boilerplate, or campaign-type labels.\n"
            "- Keywords are topics to cover naturally — do not insert raw keyword strings "
            "as awkward mid-sentence clauses or run-on SEO phrases.\n"
            "- If a keyword reads like an instruction or channel brief, ignore it."
        )

    @staticmethod
    def _grounding_rules(brand_name: str = "") -> str:
        brand = (brand_name or "the brand").strip()
        return (
            f"- Do NOT invent facts about {brand}: no fabricated case studies, win counts, "
            "client names, completed engagements, proprietary frameworks, or \"we routinely…\" "
            "performance claims unless they appear in RESEARCH STATS / CITATIONS / brand inputs.\n"
            "- Do NOT invent statistics, report titles, or attributed figures "
            "(McKinsey, GSMA, Ericsson, etc.). Use only numbers present in RESEARCH STATS / "
            "CITATIONS; if none are available, write without numeric claims.\n"
            "- Prefer hedged qualitative language over unsupported precision."
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
        # Drop prompt/meta fragments that the model would otherwise paste into prose.
        primary_keywords = self._filter_placeable_keywords(primary_keywords, limit=4)
        secondary_keywords = self._filter_placeable_keywords(secondary_keywords, limit=8)
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

        # Never keep an empty model response — retry once, then fall back to previous draft.
        if not (draft or "").strip():
            logger.warning(
                "Writer returned empty draft — retrying once | content_type=%s",
                content_type,
            )
            if content_type in SHORT_FORM_TYPES:
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
                    rewrite_instruction=(
                        rewrite_instruction
                        or "Write the complete article from scratch. Do not return empty output."
                    ),
                    primary_keywords=primary_keywords,
                    secondary_keywords=secondary_keywords,
                    target_words=target_words,
                    primary_topic=topic_lock,
                    additional_instructions=additional_instructions,
                )

        if not (draft or "").strip() and previous_draft.strip():
            logger.warning(
                "Writer still empty after retry — keeping previous draft (%d words)",
                len(previous_draft.split()),
            )
            draft = previous_draft

        # Deterministic quality boost: ensure attributed research stats are present.
        if content_type in LONG_FORM_TYPES and target_words >= 400 and (draft or "").strip():
            draft = self._enrich_factual_grounding(
                draft=draft,
                research_ctx=research_ctx,
                outline=outline,
                secondary_keywords=secondary_keywords,
            )

        # Strip common AI-cliché openers that models still insert despite prompts.
        draft = self._strip_ai_cliches(draft or "")

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

    @staticmethod
    def _resolve_audience(strategy: Dict, brand_context: Dict) -> str:
        """
        Prefer brand reader_segment so Writer never invents a wrong audience
        when strategy.audience is stale/empty.
        """
        audience = (
            brand_context.get("reader_segment")
            or strategy.get("audience")
            or []
        )
        if isinstance(audience, list):
            return ", ".join(str(a) for a in audience if str(a).strip()) or "the target audience"
        text = str(audience).strip()
        return text or "the target audience"

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

        audience_str = self._resolve_audience(strategy, brand_context)

        return ContentOutline(
            title=strategy.get("title", ""),
            content_angle=strategy.get("content_angle", ""),
            audience=audience_str,
            tone=strategy.get("tone") or brand_context.get("tone") or "professional",
            cta=strategy.get("cta") or brand_context.get("cta") or "",
            sections=sections,
            brand_name=self._brand_display_name(brand_context),
            awareness_first=self._is_awareness_first(brand_context),
        )

    def _generate_outline(
        self,
        user_input: str,
        strategy: Dict,
        brand_context: Dict,
        content_type: str,
    ) -> ContentOutline:
        """Generate a full ContentOutline using the OpenAI model."""
        target_words = WORD_COUNT_TARGETS.get(content_type, 1800)
        n_sections = "2–4" if content_type in SHORT_FORM_TYPES else "4–7"

        audience_str = self._resolve_audience(strategy, brand_context)
        tone = strategy.get("tone") or brand_context.get("tone") or "professional"
        primary_keywords = self._filter_placeable_keywords(
            strategy.get("keywords") or brand_context.get("keyword_direction", []),
            limit=3,
        )
        secondary_keywords = (
            []
            if content_type in SHORT_FORM_TYPES
            else self._filter_placeable_keywords(
                strategy.get("secondary_keywords") or [],
                limit=6,
            )
        )
        pain_points = strategy.get("pain_points") or brand_context.get("pain_points", [])

        primary_str = ", ".join(primary_keywords) or "none"
        secondary_str = ", ".join(secondary_keywords) or "none"
        brand_name = self._brand_display_name(brand_context)
        awareness_first = self._is_awareness_first(brand_context)

        awareness_outline_rules = ""
        if awareness_first:
            awareness_outline_rules = f"""
- Sections flow: empathy/problem → practical education → actionable framework → brand solution (late) → close toward CTA
- Do NOT put "{brand_name or 'the brand'}" in the first half of section headings or briefs
- Brand/product section(s) only in the final 1–2 body sections (before the reader reaches the CTA)
- Early section briefs must teach and support the reader; they must not be feature pitches
"""
        else:
            awareness_outline_rules = """
- Sections flow: problem → solution → proof → CTA
"""

        keyword_assign_rule = (
            "- Do NOT assign secondary keywords to sections; keep keywords empty or use at most one primary phrase"
            if content_type in SHORT_FORM_TYPES
            else (
                "- At least one H2 heading should include a primary or secondary keyword\n"
                "- Assign each section 1 primary-or-secondary keyword in \"keywords\" (do not invent new ones)"
            )
        )

        prompt = f"""Create a detailed content outline for a {content_type}.

USER QUERY      : {user_input}
BRAND           : {brand_name}
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
{awareness_outline_rules}- H1 title MUST include the first primary keyword naturally
{keyword_assign_rule}
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

        audience_str = self._resolve_audience(strategy, brand_context)

        return ContentOutline(
            title=str(data.get("title", "")),
            content_angle=str(data.get("content_angle", "")),
            audience=audience_str,
            tone=strategy.get("tone") or brand_context.get("tone") or "professional",
            cta=strategy.get("cta") or brand_context.get("cta") or "",
            sections=sections,
            brand_name=self._brand_display_name(brand_context),
            awareness_first=self._is_awareness_first(brand_context),
        )

    def _fallback_outline(
        self,
        user_input: str,
        strategy: Dict,
        brand_context: Dict,
    ) -> ContentOutline:
        """Rule-based fallback when LLM outline generation fails."""
        audience_str = self._resolve_audience(strategy, brand_context)
        brand_name = self._brand_display_name(brand_context)
        awareness_first = self._is_awareness_first(brand_context)
        if awareness_first:
            sections = [
                ContentSection(
                    "The Real Challenge Families Face",
                    2,
                    "Name the emotional and practical problem without pitching a product.",
                    [],
                ),
                ContentSection(
                    "What Usually Goes Wrong",
                    2,
                    "Explain common gaps in informal approaches with empathy.",
                    [],
                ),
                ContentSection(
                    "What a Solid Plan Actually Includes",
                    2,
                    "Practical checklist/framework readers can use independently.",
                    [],
                ),
                ContentSection(
                    "How a Structured Provider Helps",
                    2,
                    f"Late, concrete example of how {brand_name or 'a verified provider'} supports the plan.",
                    [],
                ),
                ContentSection(
                    "Next Steps With Confidence",
                    2,
                    "Close the educational loop; soft path toward the CTA.",
                    [],
                ),
            ]
        else:
            sections = [
                ContentSection("The Core Challenge", 2, "Define the problem and why it matters.", []),
                ContentSection("Why Existing Approaches Fall Short", 2, "Gaps in current solutions.", []),
                ContentSection("The Solution", 2, "Practical approach with concrete steps.", []),
                ContentSection("Key Benefits & Outcomes", 2, "Measurable results readers can expect.", []),
                ContentSection("Getting Started", 2, "Actionable first steps for the reader.", []),
            ]
        return ContentOutline(
            title=user_input[:80],
            content_angle="Practical guide",
            audience=audience_str,
            tone=strategy.get("tone") or brand_context.get("tone") or "professional",
            cta=strategy.get("cta") or brand_context.get("cta") or "",
            sections=sections,
            brand_name=brand_name,
            awareness_first=awareness_first,
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

        extra_block = self._format_editorial_intent(additional_instructions)

        # Micro / short long-form when user asks for very few words
        length_rules = (
            f"TARGET LENGTH   : ~{target_words} words — hit this length closely"
            if target_words < 400
            else f"TARGET LENGTH   : ~{target_words} words"
        )

        awareness_block = ""
        if outline.awareness_first:
            awareness_block = self._awareness_first_rules(outline.brand_name, outline.cta)

        prompt = f"""Write a complete {content_type} in Markdown.
{revision_block}{topic_block}{extra_block}{awareness_block}
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
{self._no_prompt_leak_rules()}
{self._grounding_rules(outline.brand_name)}
- Stay strictly on the PRIMARY TOPIC LOCK — never invert victims/roles or change the subject
- Write a hook-driven introduction (100–150 words, no heading under the H1) unless target length is under 400 words — then keep intro proportional
- Cover every outline section as `##` headings (scale section length to hit ~{target_words} words total)
- Complete every sentence — never stop mid-word or mid-sentence
- End with `## Conclusion` that recaps and closes with the exact CTA: {outline.cta}
- Prefer CTA wording like "Book an AI Discovery Call" style specificity — do not use vague "reach out today"
- When RESEARCH STATS lists any items and target length >= 400, embed at least 3 attributed statistics in the article body
  (intro or early body, one mid-article, one in proof/closing). Format: "According to <Source> (Year if available): <figure>…"
{self._stat_context_rules()}
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

        awareness_block = ""
        if outline.awareness_first:
            awareness_block = self._awareness_first_rules(outline.brand_name, outline.cta)

        prompt = f"""Revise the existing {content_type} Markdown. Do NOT rewrite from scratch.

EDITOR FEEDBACK (must fix):
{rewrite_instruction}

Preserve the overall structure, headings, and voice. Make targeted edits only.
{awareness_block}
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
{self._stat_context_rules()}
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
        awareness_line = ""
        if outline.awareness_first:
            brand = outline.brand_name or "the brand"
            awareness_line = (
                f"6. Keep awareness-first pacing: if {brand} appears in the introduction "
                "or early body, move that pitch to a late body section; do not turn the "
                "piece into a sales brochure.\n"
            )
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
{self._stat_context_rules()}
2. If listed, weave these secondary keywords naturally into intro and/or conclusion: {secondary_line}
3. Do not invent figures, organisations, or years.
4. Do not use the $ character — write USD amounts.
5. Keep structure/headings; return the FULL revised Markdown only.
{awareness_line}
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
        primary = self._filter_placeable_keywords(primary_keywords, limit=2)
        # Short-form: do not push secondary keywords into the prompt (avoids stuffing).
        secondary: List[str] = []

        topic_block = ""
        if primary_topic:
            topic_block = (
                f"\nPRIMARY TOPIC LOCK (do not change meaning/roles):\n{primary_topic}\n"
            )
        extra_block = self._format_editorial_intent(additional_instructions)
        rewrite_block = ""
        if rewrite_instruction.strip():
            rewrite_block = (
                "\nREVISION NOTES (apply meaning only — do not paste this text into the draft):\n"
                f"{rewrite_instruction.strip()}\n"
            )

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
- Include the first primary keyword early and naturally if it fits
- Do not use secondary keywords in short-form posts/emails
- Do not keyword-stuff
- Stay strictly on the primary topic; never divert or invert roles
{self._no_prompt_leak_rules()}
{self._grounding_rules(outline.brand_name)}

Human voice (important):
- Sound like a real person, not AI. Vary sentence length, use contractions, be specific
- Avoid clichés: no "in today's fast-paced world", "moreover", "furthermore", "in conclusion", "dive in", "game-changer", "unlock the power"

Write the complete {content_type}:
"""
        max_tok = 512 if target_words <= 50 else 2048
        return self._call_llm(
            system=self._system_prompt(
                outline, rewrite_instruction, primary_topic, long_form=False
            ),
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
        long_form: bool = True,
    ) -> str:
        """Shared system prompt that frames the LLM as a focused, human-sounding writer."""
        base = (
            f"You are an experienced human content writer specialising in {outline.tone.lower()} writing "
            f"for {outline.audience}. "
            "You follow formatting instructions exactly, never add meta-commentary, "
            "and return only the requested content — no preamble, no sign-off. "
            "Never invent abusive, discriminatory, or illegal how-to content. "
            "Never divert from the user's primary topic or invert roles/meaning. "
            "Never paste editorial-intent blocks, platform/format labels, or raw SEO "
            "instruction text into the published draft.\n\n"
            + self._human_voice_guide(long_form=long_form)
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
    def _human_voice_guide(long_form: bool = True) -> str:
        """Balanced, brand-safe rules that make output read as human-written."""
        secondary_line = ""
        if long_form:
            secondary_line = (
                "- Place at least one secondary keyword naturally in the introduction "
                "and one in the conclusion.\n"
            )
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
            "- Never start a sentence with Moreover, Furthermore, Additionally, or In conclusion.\n"
            f"{secondary_line}"
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
    # OpenAI wrapper
    # ------------------------------------------------------------------

    def _call_llm(
        self,
        system: str,
        user: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Invoke the OpenAI model and return plain text (retry once if empty)."""
        last_text = ""
        for attempt in range(2):
            response = self._openai.chat.completions.create(
                model=self._model,
                max_tokens=max_tokens or self._max_tokens,
                temperature=self._temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            choice = response.choices[0]
            text = (choice.message.content or "").strip()
            finish = getattr(choice, "finish_reason", None)
            if text:
                return text
            logger.warning(
                "OpenAI returned empty content | attempt=%d | finish_reason=%s",
                attempt + 1,
                finish,
            )
            last_text = text
        return last_text

    @staticmethod
    def _strip_ai_cliches(draft: str) -> str:
        """
        Deterministic cleanup of common AI-tell openers/transitions.
        Does not rewrite meaning — only removes/replaces stock phrases.
        """
        if not draft:
            return draft
        replacements = [
            (r"(?i)\bMoreover,\s*", ""),
            (r"(?i)\bFurthermore,\s*", ""),
            (r"(?i)\bAdditionally,\s*", ""),
            (r"(?i)\bIn conclusion,\s*", ""),
            (r"(?i)\bIn summary,\s*", ""),
            (r"(?i)\bTo sum up,\s*", ""),
            (r"(?i)\bIt'?s worth noting that\s*", ""),
            (r"(?i)\bIt is worth noting that\s*", ""),
            (r"(?i)\bIt'?s important to note that\s*", ""),
            (r"(?i)\bIt is important to note that\s*", ""),
            (r"(?i)\bIn today'?s fast-paced world,?\s*", ""),
            (r"(?i)\bIn today'?s digital age,?\s*", ""),
            (r"(?i)\bWhen it comes to\s+", "For "),
            (r"(?i)\bAt the end of the day,?\s*", ""),
            (r"(?i)\bNeedless to say,?\s*", ""),
        ]
        text = draft
        for pattern, repl in replacements:
            text = re.sub(pattern, repl, text)
        # Clean doubled spaces left by removals (preserve newlines)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    # ------------------------------------------------------------------
    # String utility
    # ------------------------------------------------------------------

    @staticmethod
    def _tail(text: str, n_words: int) -> str:
        """Return the last n_words of text for narrative continuity context."""
        words = text.split()
        return " ".join(words[-n_words:]) if len(words) > n_words else text
