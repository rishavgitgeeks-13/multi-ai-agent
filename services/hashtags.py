"""
Hashtag Service
===============

Generates platform-optimised hashtags for the content strategy.

Input:
    user_input    : str
    research_data : Dict   — documents, sources, statistics, citations
    brand_context : Dict   — display_name, tone, reader_segment, keyword_direction
    seo_blueprint : Dict   — output from SEOService (primary_keywords, search_intent …)
    platform      : str    — linkedin | instagram | x | website
    max_hashtags  : int    — hard cap on returned hashtags

Output: List[str]   e.g. ["#AIMarketing", "#ContentAutomation", "#SaaSGrowth"]

Pipeline:
    1. Build seed keywords from SEO blueprint + brand keyword direction
    2. Call LLM to generate platform-specific, categorised hashtag candidates
    3. Parse and deduplicate the LLM response
    4. Apply platform limits and return the final list

This service performs one LLM call and is otherwise stateless.
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional

from anthropic import Anthropic

from config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Platform-specific hashtag limits
# ---------------------------------------------------------------------------

PLATFORM_LIMITS: Dict[str, int] = {
    "linkedin": 10,
    "instagram": 30,
    "x": 3,
    "twitter": 3,
    "website": 5,
    "blog": 5,
    "email": 0,   # hashtags have no value in email
}

DEFAULT_LIMIT = 10


class HashtagService:
    """Generates a ranked, platform-optimised hashtag list."""

    def __init__(self) -> None:
        self._anthropic = Anthropic()
        self._model = settings.ANTHROPIC_MODEL
        self._temperature = settings.DEFAULT_TEMPERATURE
        logger.info("HashtagService ready | model=%s", self._model)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        user_input: str,
        research_data: Dict,
        brand_context: Dict,
        seo_blueprint: Dict,
        platform: str = "linkedin",
        max_hashtags: Optional[int] = None,
    ) -> List[str]:
        """Generate and return a deduplicated, platform-capped hashtag list."""
        logger.info("HashtagService.run() | platform=%s | query=%s…", platform, user_input[:60])

        platform_key = platform.lower().strip()

        # Email has no use for hashtags — return early
        if platform_key == "email":
            return []

        cap = max_hashtags or PLATFORM_LIMITS.get(platform_key, DEFAULT_LIMIT)
        seed_keywords = self._build_seed_keywords(seo_blueprint, brand_context)

        raw_hashtags = self._generate_via_llm(
            user_input=user_input,
            seed_keywords=seed_keywords,
            brand_context=brand_context,
            seo_blueprint=seo_blueprint,
            platform=platform_key,
            cap=cap,
        )

        final = self._deduplicate_and_cap(raw_hashtags, cap)
        logger.info("HashtagService complete | platform=%s | count=%d", platform_key, len(final))
        return final

    # ------------------------------------------------------------------
    # Step 1 — Seed keyword assembly
    # ------------------------------------------------------------------

    def _build_seed_keywords(
        self,
        seo_blueprint: Dict,
        brand_context: Dict,
    ) -> List[str]:
        """
        Combine SEO primary/secondary keywords with brand keyword direction
        as seed input for the LLM prompt.
        """
        seeds: List[str] = []

        for kw in seo_blueprint.get("primary_keywords", []):
            if kw and str(kw).strip():
                seeds.append(str(kw).strip())

        for kw in seo_blueprint.get("secondary_keywords", []):
            if kw and str(kw).strip():
                seeds.append(str(kw).strip())

        for kw in brand_context.get("keyword_direction", []):
            if kw and str(kw).strip():
                seeds.append(str(kw).strip())

        # Deduplicate while preserving order
        seen = set()
        unique: List[str] = []
        for k in seeds:
            normalised = k.lower()
            if normalised not in seen:
                seen.add(normalised)
                unique.append(k)

        return unique[:20]   # cap seeds to keep prompt concise

    # ------------------------------------------------------------------
    # Step 2 — LLM hashtag generation
    # ------------------------------------------------------------------

    def _generate_via_llm(
        self,
        user_input: str,
        seed_keywords: List[str],
        brand_context: Dict,
        seo_blueprint: Dict,
        platform: str,
        cap: int,
    ) -> List[str]:
        """Call the Anthropic model and return a raw hashtag list."""
        prompt = self._build_prompt(
            user_input=user_input,
            seed_keywords=seed_keywords,
            brand_context=brand_context,
            seo_blueprint=seo_blueprint,
            platform=platform,
            cap=cap,
        )
        try:
            response = self._anthropic.messages.create(
                model=self._model,
                max_tokens=512,
                temperature=self._temperature,
                system=(
                    "You are a social media and SEO expert. "
                    "You generate precise, high-performing hashtags. "
                    "Return valid JSON only — no prose, no markdown fences."
                ),
                messages=[{"role": "user", "content": prompt}],
            )
            return self._parse_response(response.content[0].text)
        except Exception as exc:
            logger.error("HashtagService LLM call failed: %s — using seed fallback", exc)
            return self._fallback_hashtags(seed_keywords, cap)

    def _build_prompt(
        self,
        user_input: str,
        seed_keywords: List[str],
        brand_context: Dict,
        seo_blueprint: Dict,
        platform: str,
        cap: int,
    ) -> str:
        """Construct the hashtag generation prompt."""
        brand_name = brand_context.get("display_name") or brand_context.get("brand", "")
        tone = brand_context.get("tone", "professional")
        audience = brand_context.get("reader_segment", [])
        audience_str = ", ".join(str(a) for a in audience) if audience else "general"
        search_intent = seo_blueprint.get("search_intent", "Informational")
        seeds_str = ", ".join(seed_keywords) if seed_keywords else "none"

        platform_notes = {
            "linkedin": "Professional tone. Mix: 2-3 broad industry tags + 3-4 niche topic tags + 1-2 audience tags.",
            "instagram": "Mix broad (100k+), mid-range (10k–100k), and niche (<10k) hashtags for maximum reach.",
            "x": "2-3 concise, trending hashtags. Brevity is critical.",
            "website": "SEO-relevant tags used for content categorisation.",
            "blog": "SEO-relevant tags used for content categorisation.",
        }.get(platform, "Relevant, professional hashtags.")

        return f"""Generate hashtags for a content piece.

CONTENT TOPIC   : {user_input}
BRAND           : {brand_name}
PLATFORM        : {platform}
TONE            : {tone}
AUDIENCE        : {audience_str}
SEARCH INTENT   : {search_intent}
SEED KEYWORDS   : {seeds_str}

PLATFORM NOTES  : {platform_notes}

Generate exactly {cap} hashtags, ranked by relevance (most relevant first).

Mix the following categories:
  - Industry/topic hashtags  (directly about the subject)
  - Audience hashtags        (who the content is for)
  - Trending/broad hashtags  (widely followed, high reach)

Rules:
  - CamelCase for readability: #AIMarketing not #aimarketing
  - No spaces, no punctuation inside the hashtag
  - No duplicate concepts
  - Do NOT include the brand name as a hashtag

Return ONLY this JSON object:
{{
  "hashtags": ["#Tag1", "#Tag2", ...]
}}
"""

    def _parse_response(self, raw: str) -> List[str]:
        """Parse the LLM JSON response into a clean hashtag list."""
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
        try:
            data = json.loads(cleaned)
            tags = data.get("hashtags", [])
            return [self._normalise_tag(t) for t in tags if t and str(t).strip()]
        except json.JSONDecodeError as exc:
            logger.warning("Hashtag JSON parse error: %s | raw=%s", exc, cleaned[:200])
            # Try to extract hashtags directly with regex as last resort
            return re.findall(r"#[A-Za-z][A-Za-z0-9_]+", raw)

    # ------------------------------------------------------------------
    # Step 3 — Deduplication and capping
    # ------------------------------------------------------------------

    def _deduplicate_and_cap(self, hashtags: List[str], cap: int) -> List[str]:
        """Remove duplicates (case-insensitive) and enforce the platform cap."""
        seen: set = set()
        result: List[str] = []
        for tag in hashtags:
            key = tag.lower()
            if key not in seen and tag.startswith("#"):
                seen.add(key)
                result.append(tag)
            if len(result) >= cap:
                break
        return result

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    def _fallback_hashtags(self, seed_keywords: List[str], cap: int) -> List[str]:
        """Convert seed keywords into hashtags when LLM call fails."""
        tags: List[str] = []
        for kw in seed_keywords:
            tag = "#" + re.sub(r"[^a-zA-Z0-9]", "", kw.title().replace(" ", ""))
            if len(tag) > 1:
                tags.append(tag)
        return tags[:cap]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_tag(tag: str) -> str:
        """Ensure the tag starts with # and contains no spaces."""
        tag = str(tag).strip()
        if not tag.startswith("#"):
            tag = "#" + tag
        # Remove spaces inside the tag
        tag = "#" + re.sub(r"\s+", "", tag[1:])
        return tag
