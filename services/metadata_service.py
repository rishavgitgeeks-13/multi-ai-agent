"""
Metadata Service
================

Derives and assembles the metadata block for a content piece.

Input:
    draft    : str   — raw Markdown draft from WriterService
    strategy : Dict  — title, keywords, content_type, platform, tone,
                       seo (optional), cta

Output: Dict
    {
        "title"               : str,
        "slug"                : str,
        "meta_title"          : str,
        "meta_description"    : str,
        "primary_keywords"    : List[str],
        "content_type"        : str,
        "platform"            : str,
        "language"            : str,
        "word_count"          : int,
        "reading_time_minutes": int,
        "paragraph_count"     : int,
        "heading_count"       : int,
        "headings"            : List[str],
        "has_statistics"      : bool,
        "has_lists"           : bool,
        "has_code_blocks"     : bool,
    }

This service performs no LLM calls.
All values are derived from the draft text and strategy dict.
"""

import logging
import re
import unicodedata
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# Average adult reading speed (words per minute)
_READING_SPEED_WPM = 250

# Patterns that indicate a statistic is present in the text
_STAT_PATTERNS = [
    r"\d+\s*%",           # percentage: 40%, 3.5 %
    r"\d+x\b",            # multiplier: 3x, 10x
    r"\$[\d,]+",          # dollar amounts
    r"\b\d+\s*(?:million|billion|thousand)\b",
    r"\b(?:study|report|survey|research|data)\s+(?:shows?|found|reveals?)\b",
]
_STAT_REGEX = re.compile("|".join(_STAT_PATTERNS), re.IGNORECASE)


class MetadataService:
    """Derives content metadata entirely from the draft text and strategy dict."""

    def run(self, draft: str, strategy: Dict) -> Dict:
        """Compute and return the metadata block."""
        logger.info("MetadataService.run() | draft_length=%d", len(draft))

        title = self._extract_title(draft, strategy)
        slug = self._make_slug(title)
        headings = self._extract_headings(draft)
        paragraphs = self._count_paragraphs(draft)
        word_count = self._count_words(draft)
        reading_time = self._reading_time(word_count)
        seo = strategy.get("seo", {})
        primary_keywords = self._resolve_keywords(strategy)

        metadata = {
            "title": title,
            "slug": slug,
            "meta_title": (
                seo.get("meta_title")
                or strategy.get("meta_title")
                or self._derive_meta_title(title, primary_keywords)
            ),
            "meta_description": (
                seo.get("meta_description")
                or strategy.get("meta_description")
                or self._derive_meta_description(draft, primary_keywords)
            ),
            "primary_keywords": primary_keywords,
            "content_type": str(strategy.get("content_type", "article")).lower(),
            "platform": str(strategy.get("platform", "website")).lower(),
            "language": str(strategy.get("language", "English")),
            "word_count": word_count,
            "reading_time_minutes": reading_time,
            "paragraph_count": paragraphs,
            "heading_count": len(headings),
            "headings": headings,
            "has_statistics": self._has_statistics(draft),
            "has_lists": self._has_lists(draft),
            "has_code_blocks": self._has_code_blocks(draft),
        }

        logger.info(
            "MetadataService complete | words=%d | read_time=%dm | headings=%d",
            word_count,
            reading_time,
            len(headings),
        )
        return metadata

    # ------------------------------------------------------------------
    # Title extraction
    # ------------------------------------------------------------------

    def _extract_title(self, draft: str, strategy: Dict) -> str:
        """Return the H1 title from the draft, falling back to strategy."""
        match = re.search(r"^#\s+(.+)$", draft, re.MULTILINE)
        if match:
            return match.group(1).strip()
        return str(strategy.get("title", "")).strip() or "Untitled"

    # ------------------------------------------------------------------
    # Slug generation
    # ------------------------------------------------------------------

    def _make_slug(self, title: str) -> str:
        """Convert a title into a URL-safe lowercase slug."""
        text = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii")
        text = re.sub(r"[^\w\s-]", "", text.lower())
        text = re.sub(r"[\s_]+", "-", text).strip("-")
        text = re.sub(r"-{2,}", "-", text)
        return text[:80]

    # ------------------------------------------------------------------
    # Meta title / description derivation (no LLM — rule-based fallback)
    # ------------------------------------------------------------------

    def _derive_meta_title(self, title: str, keywords: List[str]) -> str:
        """
        Build a meta title from the H1 title.
        Truncates to 60 characters; appends the top keyword if there is room.
        """
        if len(title) <= 60:
            return title
        # Truncate at the last word boundary before 60 chars
        truncated = title[:57].rsplit(" ", 1)[0]
        return truncated + "…"

    def _derive_meta_description(self, draft: str, keywords: List[str]) -> str:
        """
        Extract the first non-heading, non-empty paragraph from the draft
        as the meta description and trim to 155 characters.
        """
        lines = draft.splitlines()
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and len(stripped) > 40:
                if len(stripped) <= 155:
                    return stripped
                # Truncate at last sentence or word boundary
                trimmed = stripped[:152].rsplit(".", 1)
                return (trimmed[0] + ".") if len(trimmed) > 1 else stripped[:152] + "…"
        return ""

    # ------------------------------------------------------------------
    # Keyword resolution
    # ------------------------------------------------------------------

    def _resolve_keywords(self, strategy: Dict) -> List[str]:
        """
        Return primary keywords from the SEO blueprint if present;
        fall back to strategy keywords / brand keyword_direction.
        """
        seo = strategy.get("seo", {})
        if seo.get("primary_keywords"):
            return list(seo["primary_keywords"])
        raw = strategy.get("keywords") or strategy.get("keyword_direction", [])
        if isinstance(raw, list):
            return [str(k).strip() for k in raw if str(k).strip()]
        return []

    # ------------------------------------------------------------------
    # Heading extraction
    # ------------------------------------------------------------------

    def _extract_headings(self, draft: str) -> List[str]:
        """Return all H1–H3 heading texts in document order."""
        return [
            match.group(2).strip()
            for match in re.finditer(r"^(#{1,3})\s+(.+)$", draft, re.MULTILINE)
        ]

    # ------------------------------------------------------------------
    # Word / paragraph / reading-time metrics
    # ------------------------------------------------------------------

    def _count_words(self, draft: str) -> int:
        """Count words in the draft (strips Markdown syntax first)."""
        text = self._strip_markdown(draft)
        return len(text.split())

    def _count_paragraphs(self, draft: str) -> int:
        """Count non-empty paragraph blocks separated by blank lines."""
        blocks = re.split(r"\n{2,}", draft.strip())
        return sum(
            1 for b in blocks
            if b.strip() and not b.strip().startswith("#")
        )

    def _reading_time(self, word_count: int) -> int:
        """Return estimated reading time in minutes, minimum 1."""
        return max(1, round(word_count / _READING_SPEED_WPM))

    # ------------------------------------------------------------------
    # Content-feature flags
    # ------------------------------------------------------------------

    def _has_statistics(self, draft: str) -> bool:
        """Return True if the draft contains any statistical patterns."""
        return bool(_STAT_REGEX.search(draft))

    def _has_lists(self, draft: str) -> bool:
        """Return True if the draft contains bullet or numbered lists."""
        return bool(re.search(r"^[\s]*[-*+]\s+|^[\s]*\d+\.\s+", draft, re.MULTILINE))

    def _has_code_blocks(self, draft: str) -> bool:
        """Return True if the draft contains fenced code blocks."""
        return bool(re.search(r"```", draft))

    # ------------------------------------------------------------------
    # Markdown stripping utility
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_markdown(text: str) -> str:
        """Remove common Markdown syntax to get a clean word count."""
        # Fenced code blocks
        text = re.sub(r"```[\s\S]*?```", " ", text)
        # Inline code
        text = re.sub(r"`[^`]+`", " ", text)
        # Images
        text = re.sub(r"!\[.*?\]\(.*?\)", " ", text)
        # Links — keep link text
        text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
        # Headings
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        # Bold / italic
        text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
        text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
        # Horizontal rules
        text = re.sub(r"^[-*_]{3,}\s*$", " ", text, flags=re.MULTILINE)
        # Blockquotes
        text = re.sub(r"^>\s+", "", text, flags=re.MULTILINE)
        return text
