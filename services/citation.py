"""
Citation Service
================

Extracts, formats, and deduplicates citations from the research package.

Input:
    research_data : Dict   — documents, sources, statistics, citations
    user_input    : str

Output: List[Dict]
    [
        {
            "text"      : str,   # display / inline citation text
            "url"       : str,   # source URL (empty string if unavailable)
            "type"      : str,   # news | research | video | web | report | book
            "formatted" : str,   # full formatted citation line
        }
    ]

Pipeline:
    1. Extract structured sources from research_data["sources"]
    2. Extract plain-text citations from research_data["citations"]
    3. Format all citations with a rule-based formatter (no LLM)
    4. Merge, deduplicate, and return

This service is entirely rule-based — no LLM calls.
"""

import json
import logging
import re
from typing import Dict, List, Optional

from openai import OpenAI
from config.settings import settings

logger = logging.getLogger(__name__)

# Source type inference rules (keyword → type)
_TYPE_PATTERNS: List[tuple] = [
    (r"youtube\.com|youtu\.be", "video"),
    (r"reddit\.com", "community"),
    (r"arxiv\.org|researchgate|pubmed|scholar", "research"),
    (r"report|whitepaper|study|survey", "report"),
    (r"\.gov|\.edu", "research"),
]


class CitationService:
    """Extracts and formats citations from the research package."""

    def __init__(self) -> None:
        if not settings.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not configured."
            )

        self._openai = OpenAI(
            api_key=settings.OPENAI_API_KEY
        )

        self._model = settings.OPENAI_MODEL
        self._temperature = 0.0  # citations need deterministic output

        logger.info(
            "CitationService ready | model=%s",
            self._model,
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        research_data: Dict,
        user_input: str,
    ) -> List[Dict]:
        """Extract, format, and return all citations from the research package."""
        logger.info("CitationService.run() | query=%s…", user_input[:60])

        # Format structured source objects (rule-based)
        structured = self._format_structured_sources(
            research_data.get("sources", [])
        )

        # Format plain-text citation strings (LLM-assisted)
        plain = self._format_plain_citations(
            research_data.get("citations", []),
            user_input=user_input,
        )

        merged = self._merge_and_deduplicate(structured, plain)
        logger.info("CitationService complete | citations=%d", len(merged))
        return merged

    # ------------------------------------------------------------------
    # Step 1 — Structured sources (rule-based)
    # ------------------------------------------------------------------

    def _format_structured_sources(
        self,
        sources: List,
    ) -> List[Dict]:
        """
        Convert each source object from the research package into a
        standardised citation dict.
        Handles both dict-shaped sources and plain URL/title strings.
        """
        results: List[Dict] = []

        for source in sources:
            if isinstance(source, dict):
                citation = self._format_source_dict(source)
            elif isinstance(source, str) and source.strip():
                citation = self._format_source_string(source.strip())
            else:
                continue

            if citation:
                results.append(citation)

        return results

    def _format_source_dict(self, source: Dict) -> Optional[Dict]:
        """Format a dict-shaped source object."""
        title = str(source.get("title") or source.get("name") or "").strip()
        url = str(source.get("url") or source.get("link") or "").strip()
        author = str(source.get("author") or source.get("publisher") or source.get("source") or "").strip()
        date = str(source.get("published_date") or source.get("date") or "").strip()
        source_type = self._infer_type(url, title)

        if not title and not url:
            return None

        display = title or url
        formatted = self._build_formatted_citation(
            title=title,
            author=author,
            date=date,
            url=url,
            source_type=source_type,
        )

        return {
            "text": display,
            "url": url,
            "type": source_type,
            "formatted": formatted,
        }

    def _format_source_string(self, source: str) -> Dict:
        """Wrap a plain string source (e.g. a URL or 'Author, Title') into a citation dict."""
        url = source if source.startswith("http") else ""
        source_type = self._infer_type(url, source)
        return {
            "text": source,
            "url": url,
            "type": source_type,
            "formatted": source,
        }

    def _build_formatted_citation(
        self,
        title: str,
        author: str,
        date: str,
        url: str,
        source_type: str,
    ) -> str:
        """Build a human-readable citation line."""
        parts: List[str] = []

        if author:
            parts.append(author)
        if title:
            parts.append(f'"{title}"')
        if date:
            year = re.search(r"\b(19|20)\d{2}\b", date)
            parts.append(f"({year.group(0)})" if year else f"({date})")
        if url:
            parts.append(url)

        return ". ".join(parts) if parts else title or url

    # ------------------------------------------------------------------
    # Step 2 — Plain-text citations (LLM-assisted)
    # ------------------------------------------------------------------

    def _format_plain_citations(
        self,
        citations: List,
        user_input: str,
    ) -> List[Dict]:
        """
        Convert plain-text citation strings into structured citation dicts.
        Uses the LLM to infer type, extract metadata, and write a formatted line.
        Falls back to rule-based wrapping if the LLM call fails.
        """
        clean = [str(c).strip() for c in citations if str(c).strip()]
        if not clean:
            return []

        try:
            return [self._wrap_plain_citation(c) for c in clean]
        except Exception as exc:
            logger.warning("Citation formatting failed: %s — using empty list", exc)
            return []

    def _enrich_citations_via_llm(
        self,
        citations: List[str],
        user_input: str,
    ) -> List[Dict]:
        """Call the OpenAI model to enrich and format a list of raw citation strings."""
        numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(citations))
        prompt = f"""Format the following raw citation strings into structured citation objects.

CONTENT TOPIC: {user_input}

RAW CITATIONS:
{numbered}

For each citation, infer:
  - text      : short display name (author + year, or title)
  - url       : URL if embedded in the string, else empty string ""
  - type      : one of: news | research | video | community | report | book | web
  - formatted : clean, human-readable full citation line

Return ONLY a JSON array — one object per citation in the SAME order:
[
  {{
    "text": "McKinsey (2024)",
    "url": "",
    "type": "report",
    "formatted": "McKinsey & Company. \\"Global AI Report\\" (2024)."
  }}
]
No prose. No markdown.
"""
        response = self._openai.chat.completions.create(
            model=self._model,
            max_tokens=1024,
            temperature=self._temperature,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a citation formatter. "
                        "Return valid JSON only — no prose, no markdown fences."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        raw = response.choices[0].message.content or ""
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()

        try:
            items = json.loads(cleaned)
            return [
                {
                    "text": str(item.get("text", "")),
                    "url": str(item.get("url", "")),
                    "type": str(item.get("type", "web")),
                    "formatted": str(item.get("formatted", "")),
                }
                for item in items
                if isinstance(item, dict)
            ]
        except json.JSONDecodeError as exc:
            logger.warning("Citation JSON parse error: %s", exc)
            return [self._wrap_plain_citation(c) for c in citations]

    def _wrap_plain_citation(self, citation: str) -> Dict:
        """Rule-based fallback: wrap a plain string in a citation dict."""
        url = ""
        url_match = re.search(r"https?://\S+", citation)
        if url_match:
            url = url_match.group(0)

        return {
            "text": citation[:80],
            "url": url,
            "type": self._infer_type(url, citation),
            "formatted": citation,
        }

    # ------------------------------------------------------------------
    # Step 3 — Merge and deduplicate
    # ------------------------------------------------------------------

    def _merge_and_deduplicate(
        self,
        structured: List[Dict],
        plain: List[Dict],
    ) -> List[Dict]:
        """
        Merge both lists and remove duplicates.
        Deduplication key: normalised `formatted` string (first 60 chars).
        Structured sources take precedence over plain-text duplicates.
        """
        all_citations = structured + plain
        seen: set = set()
        unique: List[Dict] = []

        for citation in all_citations:
            key = re.sub(r"\s+", " ", citation.get("formatted", citation.get("text", ""))).strip().lower()[:60]
            if key and key not in seen:
                seen.add(key)
                unique.append(citation)

        return unique

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_type(url: str, text: str) -> str:
        """Infer the citation type from URL patterns or text keywords."""
        combined = (url + " " + text).lower()
        for pattern, source_type in _TYPE_PATTERNS:
            if re.search(pattern, combined):
                return source_type
        return "web"
