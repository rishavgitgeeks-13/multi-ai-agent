"""
Formatter Service
=================

Parses the Markdown draft into a structured, platform-ready output.

Input:
    draft    : str   — raw Markdown draft from WriterService
    strategy : Dict  — content_type, platform, keywords, tone

Output: Dict
    {
        "title"              : str,
        "markdown"           : str,          # cleaned Markdown
        "sections"           : List[Dict],   # parsed heading → content blocks
        "table_of_contents"  : List[Dict],   # [{text, level, anchor}]
        "keyword_density"    : Dict[str, float],
        "content_type"       : str,
        "platform"           : str,
    }

Each section in "sections":
    {
        "heading"    : str,
        "level"      : int,   # 1 | 2 | 3
        "anchor"     : str,   # slugified heading for TOC
        "content"    : str,   # raw text of this section
        "word_count" : int,
    }

This service performs no LLM calls.
All processing is text and regex based.
"""

import logging
import re
import unicodedata
from typing import Dict, List

logger = logging.getLogger(__name__)


class Formatter:
    """Parses and structures the Markdown draft for downstream consumers."""

    def run(self, draft: str, strategy: Dict) -> Dict:
        """Parse the draft and return the structured formatted output."""
        logger.info("Formatter.run() | draft_length=%d", len(draft))

        cleaned = self._clean_markdown(draft)
        title = self._extract_h1(cleaned)
        sections = self._parse_sections(cleaned)
        toc = self._build_toc(sections)
        keywords = self._resolve_keywords(strategy)
        keyword_density = self._compute_keyword_density(cleaned, keywords)
        content_type = str(strategy.get("content_type", "article")).lower()
        platform = str(strategy.get("platform", "website")).lower()

        result = {
            "title": title,
            "markdown": cleaned,
            "sections": sections,
            "table_of_contents": toc,
            "keyword_density": keyword_density,
            "content_type": content_type,
            "platform": platform,
        }

        logger.info(
            "Formatter complete | sections=%d | keywords=%d",
            len(sections),
            len(keywords),
        )
        return result

    # ------------------------------------------------------------------
    # Markdown cleaning
    # ------------------------------------------------------------------

    def _clean_markdown(self, draft: str) -> str:
        """
        Normalise the draft:
        - Collapse 3+ consecutive blank lines to exactly two
        - Ensure a blank line before every heading
        - Remove trailing whitespace from each line
        """
        # Strip trailing whitespace per line
        lines = [line.rstrip() for line in draft.splitlines()]
        text = "\n".join(lines)

        # Collapse excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Ensure one blank line before every heading
        text = re.sub(r"(?<!\n\n)(^#{1,3} )", r"\n\1", text, flags=re.MULTILINE)

        return text.strip()

    # ------------------------------------------------------------------
    # Title extraction
    # ------------------------------------------------------------------

    def _extract_h1(self, markdown: str) -> str:
        """Return the first H1 heading text, or empty string."""
        match = re.search(r"^#\s+(.+)$", markdown, re.MULTILINE)
        return match.group(1).strip() if match else ""

    # ------------------------------------------------------------------
    # Section parsing
    # ------------------------------------------------------------------

    def _parse_sections(self, markdown: str) -> List[Dict]:
        """
        Split the document at H1–H3 headings and return a list of sections.

        Content before the first heading is captured as a level-0 intro section.
        """
        pattern = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
        splits = list(pattern.finditer(markdown))

        sections: List[Dict] = []

        # Content before the first heading (intro / preamble)
        intro_end = splits[0].start() if splits else len(markdown)
        intro_text = markdown[:intro_end].strip()
        if intro_text:
            sections.append(self._make_section(
                heading="",
                level=0,
                content=intro_text,
            ))

        # Content under each heading
        for i, match in enumerate(splits):
            level = len(match.group(1))
            heading = match.group(2).strip()
            content_start = match.end()
            content_end = splits[i + 1].start() if i + 1 < len(splits) else len(markdown)
            content = markdown[content_start:content_end].strip()

            sections.append(self._make_section(
                heading=heading,
                level=level,
                content=content,
            ))

        return sections

    def _make_section(self, heading: str, level: int, content: str) -> Dict:
        """Build a single section dict."""
        return {
            "heading": heading,
            "level": level,
            "anchor": self._slugify(heading) if heading else "",
            "content": content,
            "word_count": len(content.split()),
        }

    # ------------------------------------------------------------------
    # Table of contents
    # ------------------------------------------------------------------

    def _build_toc(self, sections: List[Dict]) -> List[Dict]:
        """Build a flat TOC from heading sections (level 1–3, non-empty headings)."""
        return [
            {
                "text": s["heading"],
                "level": s["level"],
                "anchor": s["anchor"],
            }
            for s in sections
            if s["heading"] and s["level"] in (1, 2, 3)
        ]

    # ------------------------------------------------------------------
    # Keyword density
    # ------------------------------------------------------------------

    def _resolve_keywords(self, strategy: Dict) -> List[str]:
        """Return primary + secondary keywords for density measurement."""
        seo = strategy.get("seo", {}) or {}
        primary = (
            seo.get("primary_keywords")
            or seo.get("keywords")
            or strategy.get("keywords")
            or strategy.get("keyword_direction")
            or []
        )
        secondary = (
            seo.get("secondary_keywords")
            or strategy.get("secondary_keywords")
            or []
        )
        combined: List[str] = []
        seen = set()
        for kw in list(primary)[:2] + list(secondary)[:6]:
            normalised = str(kw).strip().lower()
            if normalised and normalised not in seen:
                seen.add(normalised)
                combined.append(normalised)
        return combined

    def _compute_keyword_density(
        self,
        markdown: str,
        keywords: List[str],
    ) -> Dict[str, float]:
        """
        Return the density (occurrences / total_words) for each keyword.
        Density is expressed as a float rounded to 4 decimal places.
        """
        if not keywords:
            return {}

        text_lower = markdown.lower()
        total_words = len(markdown.split())
        if total_words == 0:
            return {kw: 0.0 for kw in keywords}

        density: Dict[str, float] = {}
        for kw in keywords:
            # Match whole-phrase occurrences (not just substring)
            escaped = re.escape(kw)
            count = len(re.findall(r"\b" + escaped + r"\b", text_lower))
            density[kw] = round(count / total_words, 4)

        return density

    # ------------------------------------------------------------------
    # Slug utility
    # ------------------------------------------------------------------

    @staticmethod
    def _slugify(text: str) -> str:
        """Convert heading text to a lowercase, hyphen-separated anchor id."""
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
        text = re.sub(r"[^\w\s-]", "", text.lower())
        text = re.sub(r"[\s_]+", "-", text).strip("-")
        text = re.sub(r"-{2,}", "-", text)
        return text[:80]
