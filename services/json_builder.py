"""
JSON Builder Service
====================

Assembles the final output payload delivered to the caller.

Input:
    content  : Dict  — structured output from Formatter
    metadata : Dict  — metrics block from MetadataService
    strategy : Dict  — content plan (keywords, tone, cta, seo, hashtags, citations …)

Output: Dict
    {
        "status"   : "success",
        "request"  : { content_type, platform },
        "content"  : { title, markdown, sections, table_of_contents },
        "metadata" : { word_count, reading_time_minutes, … },
        "seo"      : { primary_keywords, meta_title, meta_description,
                       slug, search_intent, keyword_density },
        "hashtags" : List[str],
        "citations": List[Dict],
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

        hashtags = self._resolve_hashtags(strategy)
        citations = self._resolve_citations(strategy)
        content_block = self._build_content(content, strategy, hashtags, citations)

        final = {
            "status": "success",
            "request": self._build_request(content, strategy),
            "content": content_block,
            "metadata": self._build_metadata(metadata),
            "seo": self._build_seo(content, metadata, strategy),
            "hashtags": hashtags,
            "citations": citations,
            "cta": str(strategy.get("cta") or ""),
            "summary": self._build_summary(metadata),
            "font": str(strategy.get("font") or ""),
        }

        logger.info(
            "JSONBuilder complete | words=%d | sections=%d | citations=%d",
            metadata.get("word_count", 0),
            len(content.get("sections", [])),
            len(citations),
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

    def _build_content(
        self,
        content: Dict,
        strategy: Dict,
        hashtags: List[str],
        citations: List[Dict[str, Any]],
    ) -> Dict:
        """Project the formatter output into the content block."""
        markdown = content.get("markdown", "") or ""
        platform = str(
            content.get("platform") or strategy.get("platform") or "website"
        ).lower()
        content_type = str(
            content.get("content_type") or strategy.get("content_type") or "article"
        ).lower()

        # Append hashtags for every content type except email (if not already present).
        # Skip embedding into micro-length bodies so the word budget stays intact;
        # hashtags remain available on the payload under final_output.hashtags.
        try:
            target_n = int(strategy.get("target_word_count") or 0)
        except (TypeError, ValueError):
            target_n = 0
        micro = target_n > 0 and target_n <= 75

        if (
            hashtags
            and content_type not in ("email", "comment")
            and platform not in ("email", "comment")
            and not micro
            and not any(tag.lower() in markdown.lower() for tag in hashtags[:2])
        ):
            tag_line = " ".join(hashtags)
            markdown = f"{markdown.rstrip()}\n\nHashtags: {tag_line}\n"

        # Append Sources / References for long-form content (articles, blogs, SEO).
        # Skip email, comments, and micro posts so short formats stay clean.
        if (
            citations
            and content_type not in ("email", "comment")
            and platform not in ("email", "comment", "twitter", "x")
            and not micro
            and "## Sources" not in markdown
            and "## References" not in markdown
        ):
            lines = ["## Sources"]
            for i, cit in enumerate(citations[:12], start=1):
                label = (
                    str(cit.get("formatted") or cit.get("text") or "").strip()
                    or f"Source {i}"
                )
                url = str(cit.get("url") or "").strip()
                if url and url not in label:
                    lines.append(f"{i}. [{label}]({url})")
                elif url:
                    lines.append(f"{i}. {label}")
                else:
                    lines.append(f"{i}. {label}")
            markdown = f"{markdown.rstrip()}\n\n" + "\n".join(lines) + "\n"

        return {
            "title": content.get("title", ""),
            "markdown": markdown,
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
    # Citations
    # ------------------------------------------------------------------

    def _resolve_citations(self, strategy: Dict) -> List[Dict[str, Any]]:
        """Normalise strategy citations into a stable list of dicts."""
        raw = strategy.get("citations") or []
        if not isinstance(raw, list):
            return []

        out: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for item in raw:
            if isinstance(item, dict):
                text = str(item.get("text") or "").strip()
                url = str(item.get("url") or "").strip()
                formatted = str(item.get("formatted") or text or url).strip()
                ctype = str(item.get("type") or "web").strip() or "web"
            else:
                text = str(item or "").strip()
                url = ""
                formatted = text
                ctype = "web"
            if not formatted:
                continue
            key = (url or formatted).lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "text": text or formatted,
                    "url": url,
                    "type": ctype,
                    "formatted": formatted,
                }
            )
        return out

    # ------------------------------------------------------------------
    # Human-readable summary line
    # ------------------------------------------------------------------

    def _build_summary(self, metadata: Dict) -> str:
        """Return a one-line human-readable summary of the content piece."""
        words = metadata.get("word_count", 0)
        minutes = metadata.get("reading_time_minutes", 0)
        ct = metadata.get("content_type", "article").capitalize()
        return f"{ct} | {words} words | {minutes} min read"
