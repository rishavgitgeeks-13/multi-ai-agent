"""
Strategy Agent
==============

Transforms the research package into a complete content strategy.

Responsibilities:
    1. Run SEOService      → ranked keywords, search intent, meta fields
    2. Run HashtagService  → platform-optimised hashtag list
    3. Run CitationService → formatted citations from research sources
    4. Assemble the strategy dict consumed by the Writer Agent
    5. Persist service outputs to the shared state
    6. Route the workflow to the Writer Agent

The Strategy Agent does not generate content.
"""

import logging

from schemas.state import ContentState
from services.citation import CitationService
from services.hashtags import HashtagService
from services.seo_service import SEOService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Service instances (module-level singletons — created once, reused per call)
# ---------------------------------------------------------------------------

seo_service = SEOService()
hashtag_service = HashtagService()
citation_service = CitationService()


# ---------------------------------------------------------------------------
# Strategy node
# ---------------------------------------------------------------------------

def strategy_node(state: ContentState) -> ContentState:
    """Run all strategy services and assemble the strategy for the Writer Agent."""
    logger.info("strategy_node() | query=%s…", state["user_input"][:80])

    user_input = state["user_input"]
    research = state["research_data"]
    brand = state["brand_context"]
    platform = state.get("platform", "linkedin")
    content_type = state.get("content_type", "article")
    language = state.get("language", "English")

    # ------------------------------------------------------------------
    # 1 — SEO Service
    # ------------------------------------------------------------------
    try:
        seo_blueprint = seo_service.run(
            user_input=user_input,
            research_data=research,
            brand_context=brand,
        )
        logger.info(
            "SEOService done | primary_keywords=%d | intent=%s",
            len(seo_blueprint.get("primary_keywords", [])),
            seo_blueprint.get("search_intent", ""),
        )
    except Exception as exc:
        logger.error("SEOService failed: %s — using empty blueprint", exc)
        seo_blueprint = {
            "primary_keywords": brand.get("keyword_direction", []),
            "secondary_keywords": [],
            "keyword_scores": [],
            "search_intent": "Informational",
            "meta_title": "",
            "meta_description": "",
            "slug": "",
        }

    # ------------------------------------------------------------------
    # 2 — Hashtag Service
    # ------------------------------------------------------------------
    try:
        hashtags = hashtag_service.run(
            user_input=user_input,
            research_data=research,
            brand_context=brand,
            seo_blueprint=seo_blueprint,
            platform=platform,
        )
        logger.info("HashtagService done | hashtags=%d", len(hashtags))
    except Exception as exc:
        logger.error("HashtagService failed: %s — using empty list", exc)
        hashtags = []

    # ------------------------------------------------------------------
    # 3 — Citation Service
    # ------------------------------------------------------------------
    try:
        citations = citation_service.run(
            research_data=research,
            user_input=user_input,
        )
        logger.info("CitationService done | citations=%d", len(citations))
    except Exception as exc:
        logger.error("CitationService failed: %s — using empty list", exc)
        citations = []

    # ------------------------------------------------------------------
    # 4 — Assemble strategy dict
    # ------------------------------------------------------------------
    strategy = {
        # Core content plan
        "title": seo_blueprint.get("meta_title") or "",
        "content_angle": "",
        "audience": brand.get("reader_segment", []),
        "tone": brand.get("tone", ""),
        "outline": [],
        "cta": brand.get("cta", ""),

        # Request metadata
        "content_type": content_type,
        "platform": platform,
        "language": language,

        # Keyword strategy (from SEO service)
        "keywords": seo_blueprint.get("primary_keywords", []),
        "secondary_keywords": seo_blueprint.get("secondary_keywords", []),
        "pain_points": brand.get("pain_points", []),

        # Full SEO blueprint (consumed by Formatter + JSONBuilder)
        "seo": seo_blueprint,

        # Hashtags (consumed by JSONBuilder)
        "hashtags": hashtags,

        # Citations (consumed by Writer + JSONBuilder)
        "citations": citations,
    }

    # ------------------------------------------------------------------
    # 5 — Persist to shared state
    # ------------------------------------------------------------------
    state["strategy"] = strategy
    state["seo"] = seo_blueprint
    state["hashtags"] = hashtags

    state["current_agent"] = "strategy"
    state["next_agent"] = "writer"

    logger.info("strategy_node() complete")
    return state
