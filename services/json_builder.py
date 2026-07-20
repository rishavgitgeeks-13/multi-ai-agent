"""
JSON Builder Service
====================

Assembles the final output payload delivered to the caller.

Input:
    content  : Dict  — structured output from Formatter
    metadata : Dict  — metrics block from MetadataService
    strategy : Dict  — content plan (keywords, tone, cta, seo, hashtags …)

Output: Dict
    {
        "status"   : "success",
        "request"  : { content_type, platform },
        "content"  : { title, markdown, sections, table_of_contents },
        "metadata" : { word_count, reading_time_minutes, … },
        "seo"      : { primary_keywords, meta_title, meta_description,
                       slug, search_intent, keyword_density },
        "hashtags" : List[str],
        "cta"      : str,
        "summary"  : str,
    }

This service performs no LLM calls and no computation.
It is a pure assembly / projection layer.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class JSONBuilder:
    """Combines the formatter output, metadata, and strategy into the final payload."""

    def run(
        self,
        content: Dict,
        metadata: Dict,
        strategy: Dict,
    ) -> Dict:
        """Build and return the final output dict."""
        logger.info("JSONBuilder.run()")

        final = {
            "status": "success",
            "request": self._build_request(content, strategy),
            "content": self._build_content(content),
            "metadata": self._build_metadata(metadata),
            "seo": self._build_seo(content, metadata, strategy),
            "hashtags": self._resolve_hashtags(strategy),
            "cta": str(strategy.get("cta") or ""),
            "summary": self._build_summary(metadata),
        }

        logger.info(
            "JSONBuilder complete | words=%d | sections=%d",
            metadata.get("word_count", 0),
            len(content.get("sections", [])),
        )
        return final

    # ------------------------------------------------------------------
    # Request block
    # ------------------------------------------------------------------

    def _build_request(self, content: Dict, strategy: Dict) -> Dict:
        """Carry forward the request parameters for traceability."""
        return {
            "content_type": content.get("content_type") or strategy.get("content_type", "article"),
            "platform": content.get("platform") or strategy.get("platform", "website"),
            "tone": strategy.get("tone", ""),
            "language": strategy.get("language", "English"),
        }

    # ------------------------------------------------------------------
    # Content block
    # ------------------------------------------------------------------

    def _build_content(self, content: Dict) -> Dict:
        """Project the formatter output into the content block."""
        return {
            "title": content.get("title", ""),
            "markdown": content.get("markdown", ""),
            "sections": content.get("sections", []),
            "table_of_contents": content.get("table_of_contents", []),
        }

    # ------------------------------------------------------------------
    # Metadata block
    # ------------------------------------------------------------------

    def _build_metadata(self, metadata: Dict) -> Dict:
        """Project the metadata service output into the metadata block."""
        return {
            "word_count": metadata.get("word_count", 0),
            "reading_time_minutes": metadata.get("reading_time_minutes", 0),
            "paragraph_count": metadata.get("paragraph_count", 0),
            "heading_count": metadata.get("heading_count", 0),
            "headings": metadata.get("headings", []),
            "language": metadata.get("language", "English"),
            "has_statistics": metadata.get("has_statistics", False),
            "has_lists": metadata.get("has_lists", False),
            "has_code_blocks": metadata.get("has_code_blocks", False),
        }

    # ------------------------------------------------------------------
    # SEO block
    # ------------------------------------------------------------------

    def _build_seo(
        self,
        content: Dict,
        metadata: Dict,
        strategy: Dict,
    ) -> Dict:
        """
        Assemble the SEO block by merging:
        - strategy["seo"] (if the SEO service was run)
        - metadata (derived meta_title, meta_description, slug, keywords)
        - keyword_density from the formatter
        """
        seo_blueprint = strategy.get("seo", {})

        primary_keywords = (
            seo_blueprint.get("primary_keywords")
            or metadata.get("primary_keywords")
            or []
        )
        secondary_keywords = (
            seo_blueprint.get("secondary_keywords")
            or strategy.get("keywords")
            or []
        )

        return {
            "primary_keywords": primary_keywords,
            "secondary_keywords": secondary_keywords,
            "meta_title": (
                (seo_blueprint.get("meta_title") or "").strip()
                or (metadata.get("meta_title") or "").strip()
                or (metadata.get("title") or "Untitled")[:60]
            ),
            "meta_description": (
                (seo_blueprint.get("meta_description") or "").strip()
                or (metadata.get("meta_description") or "").strip()
                or "Read this guide for practical insights and next steps."
            ),
            "slug": (
                (seo_blueprint.get("slug") or "").strip()
                or (metadata.get("slug") or "").strip()
                or "untitled"
            ),
            "search_intent": seo_blueprint.get("search_intent", ""),
            "keyword_density": content.get("keyword_density", {}),
        }

    # ------------------------------------------------------------------
    # Hashtags
    # ------------------------------------------------------------------

    def _resolve_hashtags(self, strategy: Dict) -> List[str]:
        """Return the hashtag list from strategy, normalised."""
        raw = strategy.get("hashtags", [])
        if not isinstance(raw, list):
            return []
        return [
            ("#" + str(tag).lstrip("#").strip())
            for tag in raw
            if str(tag).strip()
        ]

    # ------------------------------------------------------------------
    # Human-readable summary line
    # ------------------------------------------------------------------

    def _build_summary(self, metadata: Dict) -> str:
        """Return a one-line human-readable summary of the content piece."""
        words = metadata.get("word_count", 0)
        minutes = metadata.get("reading_time_minutes", 0)
        ct = metadata.get("content_type", "article").capitalize()
        return f"{ct} | {words} words | {minutes} min read"
