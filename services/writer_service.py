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

SHORT_FORM_TYPES = {"linkedin", "email", "carousel", "comment"}
LONG_FORM_TYPES = {"blog", "article"}

WORD_COUNT_TARGETS: Dict[str, int] = {
    "blog": 1800,
    "article": 2200,
    "linkedin": 600,
    "email": 400,
    "carousel": 800,
    "comment": 60,
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
    font: str = ""


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
            "Do not invent missing year/scope; if absent, attribute what is available and avoid overclaiming.\n"
            "- If the PRIMARY TOPIC LOCK asks for a specific audience (e.g. NRI) or year range "
            "(e.g. 2022–2025), ONLY use stats that match that audience/range. "
            "Do NOT substitute general 'adults in India' scam percentages for NRI/property stats. "
            "If matching figures are missing from RESEARCH STATS, say so honestly in one sentence "
            "and proceed with prevention guidance — do not invent or stretch off-audience numbers.\n"
            "- Prefer .gov / major news citations over Facebook posts, social videos, or thin blogs."
        )

    @staticmethod
    def _brief_first_rules(primary_topic: str = "") -> str:
        """
        ChatGPT/Claude-style priority: user intent beats brand pitch / SEO kit defaults.
        """
        topic = (primary_topic or "").strip()
        topic_line = f"\nUSER BRIEF TO SERVE:\n{topic}\n" if topic else ""
        return f"""
BRIEF-FIRST QUALITY BAR (mandatory — write like a top assistant, not a brand brochure):
{topic_line}- Answer the user's actual ask first. Every section must earn its place against that brief.
- Do NOT substitute a generic brand essay, screening pitch, or unrelated SEO template.
- If brand CTA / keywords conflict with the brief, prefer the brief; weave brand only when it helps the reader.
- Prefer concrete, specific, useful writing over vague filler. Match the requested format (blog, post, comment, email).
- Keep geography, audience, year range, and data asks from the brief — never invent a different market.
- If research lacks an exact figure the brief asked for, say so honestly; never pad with off-topic stats.
"""

    @staticmethod
    def _cta_is_hard_required(
        content_type: str,
        primary_topic: str = "",
        awareness_first: bool = False,
        objective: str = "",
    ) -> bool:
        """Hard verbatim CTA only for commercial long-form / lead-gen intents."""
        ct = (content_type or "").lower()
        if ct in ("comment", "carousel"):
            return False
        obj = (objective or "").lower()
        if obj in ("leads", "conversion", "sales"):
            return True
        topic = (primary_topic or "").lower()
        soft_signals = (
            "explain", "what is", "what are", "guide", "how to", "tips",
            "thank", "reply", "feedback", "comment", "overview", "meaning of",
        )
        if any(s in topic for s in soft_signals):
            return False
        if awareness_first and obj in ("", "seo", "authority", "engagement", "awareness"):
            # Soft CTA in conclusion still OK, but not a hard-sell checklist item
            return False
        return ct in ("blog", "article", "email", "linkedin")

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

    # Ungrammatical SEO fragments that must not be force-inserted into prose.
    _AWKWARD_KEYWORD_RE = re.compile(
        r"("
        r"\b\w+\s+gets?\s+scammed\s+people\b|"
        r"\bhow\s+\w+\s+gets?\s+scammed\s+by\s+people\b|"
        r"\bscammed\s+people\b|"
        r"\bgets?\s+scammed\s+in\b"
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
        if cls._AWKWARD_KEYWORD_RE.search(kw):
            return True
        # Broken grammar: "X gets Y people" style fragments
        if re.search(r"\bgets?\s+\w+\s+people\b", kw, re.I):
            return True
        # Platform label as the start of a multi-word "keyword" (e.g. "linkedin announcing…")
        if len(words) >= 2 and re.match(
            r"^(linkedin|twitter|instagram|facebook|carousel|newsletter)\b",
            kw,
            re.I,
        ):
            return True
        # Length instructions must never become SEO keywords ("10 word ai blog")
        if re.search(r"\b\d{1,5}\s*[\-]?\s*words?\b", kw, re.I):
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
        Present extra guidance as hard editorial requirements.
        Do not paste the instruction text into the draft; apply it.
        """
        text = (additional_instructions or "").strip()
        if not text:
            return ""
        # Cap size so huge instruction dumps are less likely to be echoed.
        if len(text) > 2000:
            text = text[:2000].rstrip() + "…"
        return (
            "\nADDITIONAL USER REQUIREMENTS (HARD — apply all of these in the draft):\n"
            f"{text}\n"
            "Rules for this block:\n"
            "- If keyword density / primary-secondary usage is specified, hit those targets "
            "naturally (do not stuff awkwardly).\n"
            "- If hashtags are listed, include them at the end of the piece "
            "(except email / social comment formats).\n"
            "- Honour tone, structure, CTA, length, and any other constraints stated above.\n"
            "- NEVER copy this instruction block verbatim into the published draft; "
            "never mention labels like 'LINKEDIN FORMAT' or 'SEO REQUIREMENTS'.\n"
        )

    @staticmethod
    def _no_prompt_leak_rules() -> str:
        return (
            "- NEVER paste workflow meta into the draft: platform names as announcements "
            "('linkedin announcing…'), format labels, 'ADDITIONAL/EDITORIAL INTENT' text, "
            "SEO requirement boilerplate, or campaign-type labels.\n"
            "- Keywords are topics to cover naturally — do not insert raw keyword strings "
            "as awkward mid-sentence clauses or run-on SEO phrases.\n"
            "- Never force ungrammatical fragments like 'nri gets scammed people' into prose; "
            "rewrite as natural English (e.g. 'how NRIs get scammed').\n"
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
            "- Do NOT invent anonymised anecdotes or \"real-world examples\" "
            "(e.g. 'an NRI from Dubai…') unless that exact case appears in RESEARCH STATS / "
            "CITATIONS. Prefer red-flag patterns and attributed reporting instead.\n"
            "- Prefer hedged qualitative language over unsupported precision.\n"
            f"- For {brand} capability claims, stick to pain-point solutions stated in brand "
            "inputs / CTA — do not invent tech (e.g. blockchain, tamper-proof registries) "
            "unless research or brand inputs say so."
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

        topic_for_outline = (
            primary_topic or strategy.get("primary_topic") or user_input or ""
        ).strip()
        outline = self._resolve_outline(
            strategy=strategy,
            brand_context=brand_context,
            user_input=user_input,
            content_type=content_type,
            primary_topic=topic_for_outline,
        )

        research_ctx = self._build_research_context(research_data)
        rewrite_instruction = str(strategy.get("rewrite_instruction", "")).strip()
        primary_keywords, secondary_keywords = self._resolve_seo_keywords(strategy)
        # Drop prompt/meta fragments that the model would otherwise paste into prose.
        primary_keywords = self._filter_placeable_keywords(primary_keywords, limit=4)
        secondary_keywords = self._filter_placeable_keywords(secondary_keywords, limit=8)
        target_words = self._resolve_target_words(content_type, strategy)
        topic_lock = (primary_topic or strategy.get("primary_topic") or user_input or "").strip()
        # User-asked micro length must never go through the long-form article pipeline
        # (that path forces outlines, SEO sections, and 1200+ word habits).
        micro = target_words <= 75

        # On revision: surgically edit the existing draft instead of regenerating.
        if (
            rewrite_instruction
            and previous_draft.strip()
            and content_type in LONG_FORM_TYPES
            and not micro
        ):
            draft = self._revise_long_form(
                previous_draft=previous_draft,
                outline=outline,
                research_ctx=research_ctx,
                content_type=content_type,
                rewrite_instruction=rewrite_instruction,
                primary_keywords=primary_keywords,
                secondary_keywords=secondary_keywords,
                primary_topic=topic_lock,
                objective=str(strategy.get("objective") or ""),
            )
        elif micro:
            draft = self._write_micro_form(
                outline=outline,
                content_type=content_type,
                rewrite_instruction=rewrite_instruction,
                primary_keywords=primary_keywords,
                target_words=target_words,
                primary_topic=topic_lock,
                additional_instructions=additional_instructions,
                user_input=user_input,
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
                objective=str(strategy.get("objective") or ""),
            )

        # Never keep an empty model response — retry once, then fall back to previous draft.
        if not (draft or "").strip():
            logger.warning(
                "Writer returned empty draft — retrying once | content_type=%s",
                content_type,
            )
            if micro:
                draft = self._write_micro_form(
                    outline=outline,
                    content_type=content_type,
                    rewrite_instruction=rewrite_instruction
                    or "Return only the requested word count. Do not return empty output.",
                    primary_keywords=primary_keywords,
                    target_words=target_words,
                    primary_topic=topic_lock,
                    additional_instructions=additional_instructions,
                    user_input=user_input,
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
                    rewrite_instruction=(
                        rewrite_instruction
                        or "Write the complete article from scratch. Do not return empty output."
                    ),
                    primary_keywords=primary_keywords,
                    secondary_keywords=secondary_keywords,
                    target_words=target_words,
                    primary_topic=topic_lock,
                    additional_instructions=additional_instructions,
                    objective=str(strategy.get("objective") or ""),
                )

        if not (draft or "").strip() and previous_draft.strip():
            logger.warning(
                "Writer still empty after retry — keeping previous draft (%d words)",
                len(previous_draft.split()),
            )
            draft = previous_draft

        # Deterministic quality boost: ensure attributed research stats are present.
        if (
            content_type in LONG_FORM_TYPES
            and target_words >= 400
            and not micro
            and (draft or "").strip()
        ):
            draft = self._enrich_factual_grounding(
                draft=draft,
                research_ctx=research_ctx,
                outline=outline,
                secondary_keywords=secondary_keywords,
                primary_topic=topic_lock,
            )

        # Strip common AI-cliché openers that models still insert despite prompts.
        draft = self._strip_ai_cliches(draft or "")
        # Brand guideline: never ship dashes/hyphens/em-dashes in body copy.
        from services.text_cleanup import strip_all_dashes

        draft = strip_all_dashes(draft)

        # Hard length guard for micro asks (models often pad after the first line).
        if micro and (draft or "").strip():
            draft = self._enforce_micro_word_count(draft, target_words)

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
        """Prefer user-requested word count; else brand kit band; else type default."""
        user_target = strategy.get("target_word_count")
        if user_target is not None:
            try:
                n = int(user_target)
                if 1 <= n <= 50000:
                    return n
            except (TypeError, ValueError):
                pass
        # Brand SEO kit platform band (Excel) when user did not specify length
        try:
            from brands.seo_kit_loader import suggested_word_count

            kit_n = suggested_word_count(
                content_type=content_type,
                platform=str(strategy.get("platform") or "website"),
            )
            if kit_n and 1 <= int(kit_n) <= 50000:
                return int(kit_n)
        except Exception:
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
        primary_topic: str = "",
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
            primary_topic=primary_topic,
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
            font=str(brand_context.get("font") or "").strip(),
        )

    def _generate_outline(
        self,
        user_input: str,
        strategy: Dict,
        brand_context: Dict,
        content_type: str,
        primary_topic: str = "",
    ) -> ContentOutline:
        """Generate a full ContentOutline using the OpenAI model."""
        target_words = self._resolve_target_words(content_type, strategy)
        topic_lock = (
            primary_topic
            or strategy.get("primary_topic")
            or user_input
            or "topic"
        ).strip()
        if target_words <= 75:
            # Micro pieces: no multi-section outline (avoids expanding to a full article).
            return ContentOutline(
                title=topic_lock[:80],
                content_angle="concise insight",
                audience=self._resolve_audience(strategy, brand_context),
                tone=strategy.get("tone") or brand_context.get("tone") or "professional",
                cta=strategy.get("cta") or brand_context.get("cta") or "",
                sections=[],
                brand_name=self._brand_display_name(brand_context),
                awareness_first=self._is_awareness_first(brand_context),
                font=str(brand_context.get("font") or "").strip(),
            )
        n_sections = (
            "1–2"
            if target_words < 400
            else ("2–4" if content_type in SHORT_FORM_TYPES else "4–7")
        )

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
- Sections flow: answer the user brief → supporting depth → practical takeaways → soft close
- Do NOT force a sales CTA section when the brief is informational
"""

        keyword_assign_rule = (
            "- Do NOT assign secondary keywords to sections; keep keywords empty or use at most one primary phrase"
            if content_type in SHORT_FORM_TYPES
            else (
                "- At least one H2 heading should include a primary or secondary keyword "
                "ONLY when that keyword still matches the user brief\n"
                "- Assign each section 1 primary-or-secondary keyword in \"keywords\" "
                "(skip keywords that conflict with the brief; do not invent new ones)"
            )
        )

        prompt = f"""Create a detailed content outline for a {content_type}.

PRIMARY TOPIC LOCK (outline must serve this — not a brand substitute essay):
{topic_lock}

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
  "title": "<compelling H1 title that matches the PRIMARY TOPIC LOCK; include a primary keyword only if it still fits the brief>",
  "content_angle": "<unique hook that answers the user brief>",
  "sections": [
    {{
      "heading": "<section heading>",
      "heading_level": 2,
      "brief": "<1–2 sentences: what this section must cover for the user brief>",
      "keywords": ["<kw1>", "<kw2>"]
    }}
  ]
}}

Rules:
- {n_sections} sections
- Every section must advance the PRIMARY TOPIC LOCK — drop brand-template sections that do not
{awareness_outline_rules}- H1 must be grammatical English and clearly about the user brief
- Prefer natural titles over keyword-order dumps
{keyword_assign_rule}
- Headings are benefit-driven and on-brief
- Each brief is specific enough to write a full section from
- Return ONLY the JSON object — no prose, no markdown fences
"""
        try:
            raw = self._call_llm(
                system=(
                    "You are an expert content strategist. "
                    "Create precise outlines that answer the user's brief first. "
                    "Never replace the brief with a generic brand pitch outline. "
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
            font=str(brand_context.get("font") or "").strip(),
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
            font=str(brand_context.get("font") or "").strip(),
        )

    # ------------------------------------------------------------------
    # Step 3 — Research context extraction
    # ------------------------------------------------------------------

    def _build_research_context(self, research_data: Dict) -> Dict:
        """Pull statistics, citations, and reported news incidents."""
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
        incidents = [
            str(i).strip()
            for i in research_data.get("incidents", [])
            if str(i).strip()
        ]
        # Also harvest NEWS CASE lines from statistics if incidents list is empty
        if not incidents:
            incidents = [
                s.replace("NEWS CASE:", "", 1).strip()
                for s in stats
                if s.upper().startswith("NEWS CASE:")
            ]
        return {
            "stats": stats,
            "citations": citations[:_MAX_CITATIONS_GLOBAL],
            "incidents": incidents[:12],
        }

    @staticmethod
    def _format_incidents(research_ctx: Dict, n: int = 8) -> str:
        incidents = [
            str(i).strip()
            for i in (research_ctx or {}).get("incidents", [])
            if str(i).strip()
        ][:n]
        if not incidents:
            return (
                "No reported news incidents were retrieved.\n"
                "Do NOT invent city-specific abuse cases or victim stories.\n"
                "If the brief asks for cases, state that on-brief incident reporting "
                "was limited and use only attributed national context carefully."
            )
        return "\n".join(f"- {i}" for i in incidents)

    def _pick_stats(
        self,
        research_ctx: Dict,
        n: int = 3,
        primary_topic: str = "",
    ) -> str:
        """Format up to n stats; prefer ones that overlap the user brief."""
        stats = [str(s).strip() for s in research_ctx.get("stats", []) if str(s).strip()]
        if not stats:
            return (
                "No statistics are available.\n"
                "Do NOT invent percentages, benchmarks, revenue figures, "
                "survey data, or numerical claims.\n"
                "Do NOT make absolute industry claims without a citation from "
                "CITATIONS AVAILABLE; hedge or omit instead."
            )
        topic = (primary_topic or "").lower()
        tokens = [
            w
            for w in re.findall(r"[a-z0-9]{4,}", topic)
            if w
            not in {
                "write", "article", "about", "with", "from", "that", "this",
                "please", "content", "blog", "post",
            }
        ][:10]
        if tokens:
            ranked = sorted(
                stats,
                key=lambda s: sum(1 for t in tokens if t in s.lower()),
                reverse=True,
            )
            # Keep any with at least one overlap first; fill remainder in original order
            selected = [s for s in ranked if any(t in s.lower() for t in tokens)][:n]
            if len(selected) < n:
                for s in stats:
                    if s not in selected:
                        selected.append(s)
                    if len(selected) >= n:
                        break
        else:
            selected = stats[:n]
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
        objective: str = "",
    ) -> str:
        """Write the full long-form piece in a single LLM call (token-efficient)."""
        if target_words is None:
            target_words = WORD_COUNT_TARGETS.get(content_type, 1800)
        primary = [str(k) for k in (primary_keywords or []) if str(k).strip()][:2]
        secondary = [str(k) for k in (secondary_keywords or []) if str(k).strip()][:6]
        primary_str = ", ".join(primary) or "none"
        secondary_str = ", ".join(secondary) or "none"
        lead_primary = primary[0] if primary else ""
        hard_cta = self._cta_is_hard_required(
            content_type,
            primary_topic=primary_topic,
            awareness_first=outline.awareness_first,
            objective=objective,
        )
        cta_line = (outline.cta or "").strip()
        if hard_cta and cta_line:
            cta_rules = (
                f"- End with `## Conclusion` that recaps and closes with the exact CTA: {cta_line}\n"
                "- Prefer specific CTA wording — do not use vague \"reach out today\""
            )
            cta_header = f"CTA (use verbatim): {cta_line}"
            rev_cta = f"- End with the exact CTA phrase: {cta_line}"
        else:
            cta_rules = (
                "- End with `## Conclusion` that recaps the brief and gives a natural next step "
                "for the reader. Brand CTA is optional — only include it if it still fits the brief."
            )
            cta_header = (
                f"CTA (optional / soft): {cta_line or 'none — close naturally on the brief'}"
            )
            rev_cta = (
                "- Close on-brief; include brand CTA only if it still fits the user brief"
            )

        citations_block = (
            "\n".join(f"- {c}" for c in research_ctx.get("citations", [])[:8])
            or "none"
        )
        stats_n = 8
        revision_block = ""
        if rewrite_instruction:
            revision_block = f"""
CRITICAL REVISION PASS — you must apply these editor notes:
{rewrite_instruction}

Mandatory fixes for this revision (do not skip):
- Stay on the PRIMARY TOPIC LOCK. If the draft drifted into a brand pitch, rewrite back to the brief.
- Embed up to 3 on-brief statistics from RESEARCH STATS (only if they match the brief),
  each with clear attribution (source + figure/year when present).
- Never invent organisation names, report titles, years, or percentages.
- If a research snippet is vague or off-brief, omit it — do not force a stats quota.
- Remove absolute uncited claims unless they appear in stats/citations.
- Place secondary keywords only when they still fit the brief (intro + conclusion when natural).
{rev_cta}
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
        brief_block = self._brief_first_rules(primary_topic)

        length_rules = (
            f"TARGET LENGTH   : ~{target_words} words — hit this length closely"
            if target_words < 400
            else f"TARGET LENGTH   : ~{target_words} words"
        )

        awareness_block = ""
        if outline.awareness_first:
            awareness_block = self._awareness_first_rules(outline.brand_name, outline.cta)

        h1_kw_rule = (
            f'- Start with `# {outline.title}` — include "{lead_primary}" in the H1 '
            "only if it still matches the PRIMARY TOPIC LOCK"
            if lead_primary
            else f"- Start with `# {outline.title}`"
        )

        prompt = f"""Write a complete {content_type} in Markdown.
{revision_block}{topic_block}{brief_block}{extra_block}{awareness_block}
TITLE           : {outline.title}
CONTENT ANGLE   : {outline.content_angle}
AUDIENCE        : {outline.audience}
TONE            : {outline.tone}
{cta_header}
{length_rules}
PRIMARY KEYWORDS: {primary_str}
SECONDARY KEYWORDS: {secondary_str}

OUTLINE TO FOLLOW:
{self._format_section_list(outline.sections)}

RESEARCH STATS (prefer on-brief items — do not invent figures):
{self._pick_stats(research_ctx, n=stats_n, primary_topic=primary_topic)}

REPORTED NEWS INCIDENTS (evidence-driven cases from news — use when the brief asks for cases):
{self._format_incidents(research_ctx, n=8)}

CITATIONS AVAILABLE:
{citations_block}

SEO placement rules (apply without hijacking the brief):
{h1_kw_rule}
- Use primary keywords naturally in intro/body when they fit the brief (no stuffing)
- Use secondary keywords only when they still match the brief
- At least one `##` heading may contain a keyword if it remains grammatical and on-topic

Content rules:
- Write like a skilled human editor: clear, specific, useful — not generic AI filler
{self._no_prompt_leak_rules()}
{self._grounding_rules(outline.brand_name)}
- Stay strictly on the PRIMARY TOPIC LOCK — never invert victims/roles or change the subject
- Follow geography from the PRIMARY TOPIC LOCK / user brief only (India, US, UK, etc.);
  do not invent a market from the brand, and do not fill with unrelated-country forum stats
- When the brief asks for cases / incidents (e.g. nanny abuse cases), lead with REPORTED NEWS INCIDENTS
  (city, year, allegation/charges, outlet). Do NOT fill the article with only national NCRB/POCSO
  totals that are not nanny-specific. You may cite NCRB briefly as broader context and must say
  clearly when official nanny-specific aggregates or state-wise nanny counts are unavailable.
- When the user asked for numbers/cases/state-wise data, prioritize REPORTED NEWS INCIDENTS + on-brief
  RESEARCH STATS; never invent case counts or anonymous victim stories
- Write a hook-driven introduction (100–150 words, no heading under the H1) unless target length is under 400 words — then keep intro proportional
- Cover every outline section as `##` headings (scale section length to hit ~{target_words} words total)
- Complete every sentence — never stop mid-word or mid-sentence
{cta_rules}
- When RESEARCH STATS lists on-brief items and target length >= 400, embed up to 3 attributed statistics
  (intro or early body, one mid-article, one in proof/closing). Format: "According to <Source> (Year if available): <figure>…"
  Do NOT repeat the same statistic three times. Do NOT cite Facebook posts/videos as primary evidence.
  Do NOT invent stats to hit a quota when on-brief research is thin.
{self._stat_context_rules()}
- When a proof / case-study / real-world section appears in the outline, ground it with research stats or named citations above — do not use brand name alone as proof; do not invent anonymous case stories
- Never invent percentages, benchmarks, financial figures, organisation names, or report titles
- Do not state absolute industry claims (e.g. "most startups fail because…") unless that exact claim appears in RESEARCH STATS or CITATIONS AVAILABLE; otherwise hedge or omit
- Write money amounts as "USD 500" or "USD 50,000" — never use the $ character (breaks Markdown renderers)
- Match brand tone exactly throughout: {outline.tone}
- Return ONLY Markdown — no preamble

Write the complete {content_type}:
"""
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
        primary_topic: str = "",
        objective: str = "",
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
        draft_for_edit = previous_draft
        if len(draft_for_edit) > 14000:
            draft_for_edit = previous_draft[:7000] + "\n\n…\n\n" + previous_draft[-5000:]

        awareness_block = ""
        if outline.awareness_first:
            awareness_block = self._awareness_first_rules(outline.brand_name, outline.cta)

        hard_cta = self._cta_is_hard_required(
            content_type,
            primary_topic=primary_topic,
            awareness_first=outline.awareness_first,
            objective=objective,
        )
        cta_item = (
            f"5. Closing CTA must use verbatim: {outline.cta}"
            if hard_cta and (outline.cta or "").strip()
            else "5. Closing must stay on the user brief; brand CTA only if it still fits"
        )
        topic_block = (
            f"\nPRIMARY TOPIC LOCK (do not abandon on revision):\n{primary_topic}\n"
            if primary_topic
            else ""
        )

        prompt = f"""Revise the existing {content_type} Markdown. Do NOT rewrite from scratch.
{topic_block}
{self._brief_first_rules(primary_topic)}
EDITOR FEEDBACK (must fix):
{rewrite_instruction}

Preserve the overall structure, headings, and voice. Make targeted edits only.
{awareness_block}
BRAND TONE (exact) : {outline.tone}
AUDIENCE           : {outline.audience}
CTA                : {outline.cta or "optional"}
PRIMARY KEYWORDS   : {primary_str}
SECONDARY KEYWORDS : {secondary_str}

RESEARCH STATS (prefer on-brief — do not invent figures):
{self._pick_stats(research_ctx, n=8, primary_topic=primary_topic)}

REPORTED NEWS INCIDENTS:
{self._format_incidents(research_ctx, n=8)}

CITATIONS AVAILABLE:
{citations_block}

Mandatory edit checklist:
1. If the draft is off-brief, rewrite drifted sections back to the PRIMARY TOPIC LOCK first.
2. If the brief asks for cases/incidents and the draft only has generic NCRB totals, rewrite the
   cases section using REPORTED NEWS INCIDENTS (name city/outlet/year when present).
3. Insert up to 3 attributed on-brief statistics from RESEARCH STATS when available
   (do not invent or force off-brief stats to hit a quota).
{self._stat_context_rules()}
4. Place secondary keywords naturally only when they still fit the brief.
{cta_item}
6. Match tone exactly: {outline.tone}
7. Write currency as "USD 500" — never use the $ character.
8. Complete every sentence.
9. Keep strong existing on-brief sections; only edit what the editor feedback requires.

EXISTING DRAFT:
{draft_for_edit}

Return the FULL revised Markdown article only — no preamble.
"""
        return self._call_llm(
            system=self._system_prompt(outline, rewrite_instruction, primary_topic),
            user=prompt,
            max_tokens=8192,
        )

    def _enrich_factual_grounding(
        self,
        draft: str,
        research_ctx: Dict,
        outline: ContentOutline,
        secondary_keywords: Optional[List[str]] = None,
        primary_topic: str = "",
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
        topic_line = (
            f"PRIMARY TOPIC LOCK (do not drift while enriching):\n{primary_topic}\n\n"
            if primary_topic
            else ""
        )
        prompt = f"""Improve factual grounding of this Markdown article with MINIMAL edits.
{topic_line}Tone to preserve: {outline.tone}
CTA to preserve if still on-brief: {outline.cta or "optional"}

RESEARCH STATS (only use these — prefer on-brief; do not invent):
{self._pick_stats(research_ctx, n=8, primary_topic=primary_topic)}

CITATIONS:
{chr(10).join(f"- {c}" for c in research_ctx.get("citations", [])[:6]) or "none"}

Required edits:
1. Prefer on-brief attributed statistics from RESEARCH STATS (audience + year range if the brief asks).
   Embed up to 3 distinct figures — do NOT repeat the same stat, do NOT cite Facebook as primary evidence,
   and do NOT invent numbers to hit a quota. If on-brief stats are thin, keep guidance honest.
   Format: "According to <Source> (Year): <figure>…"
{self._stat_context_rules()}
2. If listed, weave these secondary keywords naturally into intro and/or conclusion: {secondary_line}
   Skip ungrammatical fragments; use natural English instead. Skip keywords that fight the brief.
3. Remove invented anonymous anecdotes / unsourced "real-world examples".
4. Do not invent figures, organisations, or years.
5. Do not use the $ character — write USD amounts.
6. Keep structure/headings; return the FULL revised Markdown only.
7. Never replace the article topic with a brand pitch while enriching.
{awareness_line}
DRAFT:
{draft[:12000]}
"""
        try:
            enriched = self._call_llm(
                system=(
                    "You are a careful editorial reviser. Stay on the user's primary topic. "
                    "Make minimal targeted edits. Return only the full Markdown article."
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
    # Step 4a — Micro writing (user asked for a tiny word count)
    # ------------------------------------------------------------------

    def _write_micro_form(
        self,
        outline: ContentOutline,
        content_type: str,
        rewrite_instruction: str = "",
        primary_keywords: Optional[List[str]] = None,
        target_words: int = 10,
        primary_topic: str = "",
        additional_instructions: str = "",
        user_input: str = "",
    ) -> str:
        """Write an ultra-short piece that must hit the exact word budget."""
        topic = (primary_topic or user_input or outline.title or "AI").strip()
        # Drop length meta from the topic so the model does not write about "10 words"
        topic = re.sub(
            r"\b(?:exactly\s+|about\s+|around\s+|~)?\d{1,5}\s*[\-]?\s*words?\b",
            " ",
            topic,
            flags=re.I,
        )
        topic = re.sub(r"\s{2,}", " ", topic).strip(" -,:;") or "AI"
        primary = self._filter_placeable_keywords(primary_keywords, limit=1)
        primary_str = primary[0] if primary else ""

        rewrite_block = ""
        if rewrite_instruction.strip():
            rewrite_block = (
                "\nREVISION: shorten or adjust to the exact word count. "
                "Do not expand.\n"
                f"{rewrite_instruction.strip()[:400]}\n"
            )
        extra = self._format_editorial_intent(additional_instructions)

        is_comment = content_type == "comment" or "comment" in topic.lower() or (
            "reply" in topic.lower() and target_words <= 40
        )
        if is_comment or content_type == "comment":
            prompt = f"""Write a social COMMENT / reply that is EXACTLY {target_words} words.

USER INTENT / POST CONTEXT: {topic}
{extra}{rewrite_block}
HARD RULES:
- EXACTLY {target_words} words. Count carefully.
- This is a reply under someone's post (e.g. thanking them for saying the article is good).
- Sound like a real person. Warm and brief.
- NO brand pitch, NO sales CTA, NO hashtags, NO stats, NO headings, NO SEO keywords.
- NO incomplete sentences. Return ONLY the comment text.
"""
            system = (
                "You write ultra-short social comments. Match the user's intent. "
                "Never pitch products. Never exceed the word count."
            )
        else:
            prompt = f"""Write a {content_type} that is EXACTLY {target_words} words. No more, no less.

TOPIC: {topic}
TONE: {outline.tone}
OPTIONAL KEYWORD (use only if it fits naturally): {primary_str or "none"}
{extra}{rewrite_block}
HARD RULES:
- Output EXACTLY {target_words} words total (count every word carefully).
- Plain prose only. No Markdown headings (# ##), no bullet lists, no hashtags.
- Do NOT mention the word count, "10 word", "word blog", or similar meta phrases.
- Do NOT invent statistics, citations, or multi-paragraph essays.
- Do NOT pad with sections, titles, or repeated CTAs.
- A short natural closing CTA is allowed ONLY if it still fits inside the {target_words}-word budget.
- Return ONLY the {target_words}-word text. Nothing else.
"""
            system = (
                "You write ultra-short branded copy. You count words precisely and "
                "never exceed the requested length. Never turn a micro request into "
                "a long article."
            )
        return self._call_llm(
            system=system,
            user=prompt,
            max_tokens=max(64, target_words * 4),
        )

    @staticmethod
    def _enforce_micro_word_count(draft: str, target_words: int) -> str:
        """Trim padded micro drafts to the requested word count."""
        if not draft or target_words <= 0:
            return draft
        # Prefer the first non-empty paragraph (models often pad after a good opener).
        chunks = [c.strip() for c in re.split(r"\n\s*\n", draft.strip()) if c.strip()]
        candidate = chunks[0] if chunks else draft.strip()
        # Strip heading markers if model ignored instructions
        candidate = re.sub(r"^#+\s*", "", candidate).strip()
        words = candidate.split()
        if len(words) > target_words:
            candidate = " ".join(words[:target_words])
        return candidate

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
        brief_block = self._brief_first_rules(primary_topic)
        extra_block = self._format_editorial_intent(additional_instructions)
        rewrite_block = ""
        if rewrite_instruction.strip():
            rewrite_block = (
                "\nREVISION NOTES (apply meaning only — do not paste this text into the draft):\n"
                f"{rewrite_instruction.strip()}\n"
            )
        soft_cta = not self._cta_is_hard_required(
            content_type,
            primary_topic=primary_topic,
            awareness_first=outline.awareness_first,
            objective="",
        )
        cta_line = (
            f"CTA (soft / optional): {outline.cta or 'none — close naturally'}"
            if soft_cta
            else f"CTA: {outline.cta}"
        )

        prompt = f"""Write a complete {content_type} for {platform}.
{topic_block}{brief_block}{extra_block}{rewrite_block}
TITLE / TOPIC   : {outline.title}
CONTENT ANGLE   : {outline.content_angle}
AUDIENCE        : {outline.audience}
TONE            : {outline.tone}
{cta_line}
TARGET LENGTH   : ~{target_words} words — adhere closely to this length
PRIMARY KEYWORDS: {", ".join(primary) or "none"}
SECONDARY KEYWORDS: {", ".join(secondary) or "none"}

CONTENT STRUCTURE TO COVER:
{self._format_section_list(outline.sections)}

RELEVANT STATS:
{self._pick_stats(research_ctx, n=3, primary_topic=primary_topic)}

FORMAT REQUIREMENTS:
{format_rules}

SEO notes:
- Include the first primary keyword early and naturally if it fits the brief (skip for comments)
- Do not use secondary keywords in short-form posts/emails/comments
- Do not keyword-stuff
- Stay strictly on the primary topic; never divert into a brand pitch if the brief is different
{self._no_prompt_leak_rules()}
{self._grounding_rules(outline.brand_name) if content_type != "comment" else "- Do not invent stats or hard-sell the brand in a comment."}

Human voice (important):
- Sound like a real person, not AI. Vary sentence length, use contractions, be specific
- Avoid clichés: no "in today's fast-paced world", "moreover", "furthermore", "in conclusion", "dive in", "game-changer", "unlock the power"

Write the complete {content_type}:
"""
        # Comments: do not push SEO keywords into the model prompt
        if content_type == "comment" or (platform or "").lower() == "comment":
            prompt = f"""Write a paste-ready social media COMMENT / reply only.
{topic_block}{extra_block}{rewrite_block}
TOPIC / POST CONTEXT: {outline.title or primary_topic or "the post"}
TONE: {outline.tone}
TARGET LENGTH: ~{target_words} words (short reply)

FORMAT REQUIREMENTS:
{format_rules}

Return ONLY the comment text. No hashtags. No titles. No lists.
"""
        max_tok = 256 if content_type == "comment" or target_words <= 50 else 2048
        return self._call_llm(
            system=(
                "You write natural social media comments/replies. "
                "Match the user's intent. Never add hashtags or promo blocks."
                if content_type == "comment" or (platform or "").lower() == "comment"
                else self._system_prompt(
                    outline, rewrite_instruction, primary_topic, long_form=False
                )
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
        """Shared system prompt — brief-first, human, ChatGPT/Claude-level usefulness."""
        base = (
            f"You are an expert editorial writer with the judgment of a senior human editor "
            f"(clear, specific, useful — {outline.tone.lower()} voice for {outline.audience}). "
            "Serve the user's brief first, like ChatGPT or Claude would: stay on topic, "
            "match the requested format, and never replace the ask with a generic brand pitch. "
            "Follow formatting instructions exactly, never add meta-commentary, "
            "and return only the requested content — no preamble, no sign-off. "
            "Never invent abusive, discriminatory, or illegal how-to content. "
            "Never divert from the user's primary topic or invert roles/meaning. "
            "Never paste editorial-intent blocks, platform/format labels, or raw SEO "
            "instruction text into the published draft.\n\n"
            + self._human_voice_guide(long_form=long_form)
        )
        if outline.font:
            base += (
                f"\n\nBRAND FONT GUIDELINE: Publish-ready copy for this brand uses "
                f"'{outline.font}'. Do not mention the font name in the article body; "
                "follow brand voice and the no-dash typography rules above."
            )
        if primary_topic:
            base += (
                f"\n\nPRIMARY TOPIC LOCK (highest priority — higher than brand CTA/SEO kit):\n"
                f"{primary_topic}"
            )
        if rewrite_instruction:
            base += (
                f"\n\nREVISION INSTRUCTIONS FROM EDITOR:\n{rewrite_instruction}\n"
                "Apply these instructions throughout the entire piece while staying on the "
                "PRIMARY TOPIC LOCK."
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
            "- Never use stiff scaffolding: \"First, discuss…\", \"In this article, we will…\", "
            "\"Let us examine…\", \"This article explores…\", \"As we delve…\".\n"
            "- Prefer plain spoken words. Say \"NRI families\" or \"families living abroad\" — "
            "never \"expatriates\". Prefer \"parents\" over \"guardians seeking premium care\" "
            "unless that exact segment is required.\n"
            "- Prefer everyday phrasing over academic paper tone (avoid long citation-title dumps "
            "as sentence subjects; attribute briefly).\n"
            f"{secondary_line}"
            "- Use contractions naturally (it's, you're, don't, we've).\n"
            "- Prefer concrete, specific nouns and real examples over vague generalities.\n"
            "- Address the reader directly with \"you\" where it fits; light first-person (\"we\") is fine.\n"
            "- Do not over-hedge or over-explain. Trust the reader.\n"
            "- Avoid formulaic scaffolding (e.g. rigidly equal sections, a forced summary that "
            "restates everything). End with a genuine, specific closing rather than a generic wrap-up.\n"
            "- Keep it professional and on-brand, natural, not slangy or unprofessional.\n"
            "- CRITICAL TYPOGRAPHY: Never use any dash characters in the draft "
            "(no hyphen -, no en dash, no em dash). Rewrite as separate words or commas "
            "(write 'well being' not 'well-being'; '2020 to 2026' not '2020-2026'). "
            "Use asterisk bullets (*) instead of dash bullets.\n"
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
        plat = (platform or "").lower().strip()
        # Platform wins when comment (even if content_type was aliased)
        if content_type == "comment" or plat == "comment":
            return (
                "- Output ONLY the comment text a user can paste under a post\n"
                "- 2 to 5 short sentences; stay under ~80 words unless user asked otherwise\n"
                "- Match the user's intent exactly (agree / insight / question / thanks)\n"
                "- If a source post was provided, reply to that post — do not invent a new post\n"
                "- NO hashtags, NO headings, NO bullet lists, NO SEO keyword stuffing\n"
                "- NO hard CTA, NO 'Hashtags:' footer, NO title line\n"
                "- Soft brand mention only if the user explicitly asked for it"
            )
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
            (r"(?i)\bexpatriates\b", "families living abroad"),
            (r"(?i)\bexpatriate\b", "family living abroad"),
            (r"(?i)\bFirst,\s+discuss\b", "Start with"),
            (r"(?i)\bFirst discuss\b", "Start with"),
            (r"(?i)\bThis article explores\b", "Here's a clear look at"),
            (r"(?i)\bIn this article,?\s+we will\b", "We'll"),
            (r"(?i)\bLet us examine\b", "Look at"),
            (r"(?i)\bAs we delve into\b", "On"),
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
