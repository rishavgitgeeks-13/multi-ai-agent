"""
Citation Validator
==================

Validates citation dicts produced by CitationService and consumed by the
Writer Agent, JSONBuilder, and final_output assembly.

Validates:
  - Individual citation dicts (required fields, type enum, URL format)
  - Batches of citations (deduplication, minimum count)
  - Source dicts from research tools (NewsSearch, TavilySearch, YouTubeSearch)

Usage
-----
    from validators.citation_validator import CitationValidator

    # Validate a single citation
    result = CitationValidator.validate_citation(citation_dict)

    # Validate a full list
    result = CitationValidator.validate_citation_list(citations)
    if result.ok:
        clean_citations = result.data

    # Validate a raw source from a research tool
    result = CitationValidator.validate_source(source_dict)
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Valid citation type values (CitationService._infer_type outputs)
VALID_CITATION_TYPES = frozenset({
    "news", "research", "video", "community", "report", "book", "web"
})

# Valid source_type values (ResearchDocument.source_type)
VALID_SOURCE_TYPES = frozenset({
    "kb", "web", "news", "youtube", "reddit"
})

# Minimum character length for a citation formatted string to be considered valid
_MIN_FORMATTED_LEN = 5


# ==========================================================================
# Result wrapper (mirrors json_validator.ValidationResult)
# ==========================================================================


@dataclass
class ValidationResult:
    ok: bool
    data: Any = None
    error: str = ""
    warnings: List[str] = field(default_factory=list)

    @classmethod
    def success(cls, data: Any, warnings: List[str] = None) -> "ValidationResult":
        return cls(ok=True, data=data, warnings=warnings or [])

    @classmethod
    def failure(cls, error: str, data: Any = None) -> "ValidationResult":
        return cls(ok=False, error=error, data=data)


# ==========================================================================
# Citation Validator
# ==========================================================================


class CitationValidator:
    """
    Validates citation and source objects at the boundary between
    CitationService / research tools and the Writer / final_output pipeline.
    """

    # ------------------------------------------------------------------
    # Single citation dict
    # ------------------------------------------------------------------

    @classmethod
    def validate_citation(cls, citation: Any) -> ValidationResult:
        """
        Validate one citation dict produced by CitationService.run().

        Expected shape:
        {
            "text"      : str,   # display / inline citation text (required)
            "url"       : str,   # source URL or empty string
            "type"      : str,   # one of VALID_CITATION_TYPES
            "formatted" : str,   # human-readable full citation line (required)
        }
        """
        if not isinstance(citation, dict):
            return ValidationResult.failure(
                f"Citation must be a dict, got {type(citation).__name__}."
            )

        warnings: List[str] = []
        clean: Dict[str, str] = {}

        # --- text (required) ---
        text = str(citation.get("text", "")).strip()
        if not text:
            return ValidationResult.failure(
                "Citation 'text' is required and cannot be empty."
            )
        clean["text"] = text

        # --- url (optional, but must be valid if provided) ---
        url = str(citation.get("url", "")).strip()
        if url:
            url_result = cls._validate_url(url)
            if not url_result.ok:
                warnings.append(f"Invalid URL '{url}': {url_result.error} — cleared.")
                url = ""
        clean["url"] = url

        # --- type (required, enum) ---
        citation_type = str(citation.get("type", "web")).strip().lower()
        if citation_type not in VALID_CITATION_TYPES:
            warnings.append(
                f"Unknown citation type '{citation_type}' — defaulting to 'web'."
            )
            citation_type = "web"
        clean["type"] = citation_type

        # --- formatted (required) ---
        formatted = str(citation.get("formatted", "")).strip()
        if not formatted or len(formatted) < _MIN_FORMATTED_LEN:
            # Fall back to constructing from available fields
            formatted = cls._build_formatted(text, url)
            warnings.append(
                f"'formatted' was missing or too short — reconstructed: '{formatted}'."
            )
        clean["formatted"] = formatted

        return ValidationResult.success(clean, warnings)

    # ------------------------------------------------------------------
    # List of citations
    # ------------------------------------------------------------------

    @classmethod
    def validate_citation_list(
        cls,
        citations: Any,
        min_count: int = 0,
        deduplicate: bool = True,
    ) -> ValidationResult:
        """
        Validate a list of citation dicts.

        Parameters
        ----------
        citations   : the raw list to validate
        min_count   : minimum number of valid citations required (0 = no minimum)
        deduplicate : if True, remove duplicate citations by normalised `formatted` key

        Returns ValidationResult with data=List[Dict] (only the valid ones).
        """
        if not isinstance(citations, list):
            return ValidationResult.failure(
                f"Citations must be a list, got {type(citations).__name__}."
            )

        warnings: List[str] = []
        valid: List[Dict] = []

        for i, citation in enumerate(citations):
            result = cls.validate_citation(citation)
            if result.ok:
                valid.append(result.data)
                warnings.extend([f"[{i}] {w}" for w in result.warnings])
            else:
                warnings.append(f"[{i}] Skipped invalid citation: {result.error}")

        if deduplicate:
            valid, dedup_warnings = cls._deduplicate(valid)
            warnings.extend(dedup_warnings)

        if min_count and len(valid) < min_count:
            return ValidationResult.failure(
                f"Only {len(valid)} valid citations found (minimum required: {min_count}).",
                data=valid,
            )

        return ValidationResult.success(valid, warnings)

    # ------------------------------------------------------------------
    # Raw source dicts (from research tools)
    # ------------------------------------------------------------------

    @classmethod
    def validate_source(cls, source: Any) -> ValidationResult:
        """
        Validate a source dict as returned by TavilySearch, NewsSearch,
        or YouTubeSearch before it is added to research_data["sources"].

        Expected minimum shape:
        {
            "title"        : str,
            "url"          : str,
            "source_type"  : str,   # kb | web | news | youtube | reddit
            "published_date": str,  # optional
            "author"       : str,   # optional
            "snippet"      : str,   # optional
        }
        """
        if not isinstance(source, dict):
            return ValidationResult.failure(
                f"Source must be a dict, got {type(source).__name__}."
            )

        warnings: List[str] = []
        clean: Dict[str, str] = {}

        # --- title ---
        title = str(source.get("title") or source.get("name") or "").strip()
        if not title:
            warnings.append("Source has no title — will use URL as display text.")
        clean["title"] = title

        # --- url ---
        url = str(source.get("url") or source.get("link") or "").strip()
        if url:
            url_result = cls._validate_url(url)
            if not url_result.ok:
                warnings.append(f"Invalid URL '{url}': {url_result.error}.")
                url = ""
        if not url and not title:
            return ValidationResult.failure(
                "Source must have at least a title or a URL."
            )
        clean["url"] = url

        # --- source_type ---
        source_type = str(source.get("source_type") or source.get("type") or "web").strip().lower()
        if source_type not in VALID_SOURCE_TYPES:
            warnings.append(
                f"Unknown source_type '{source_type}' — defaulting to 'web'."
            )
            source_type = "web"
        clean["source_type"] = source_type

        # --- optional fields ---
        clean["published_date"] = str(source.get("published_date") or source.get("published_at") or "").strip()
        clean["author"] = str(source.get("author") or source.get("channel") or "").strip()
        clean["snippet"] = str(source.get("snippet") or source.get("description") or "").strip()

        return ValidationResult.success(clean, warnings)

    @classmethod
    def validate_source_list(cls, sources: Any) -> ValidationResult:
        """Validate a list of source dicts. Returns only valid entries."""
        if not isinstance(sources, list):
            return ValidationResult.failure(
                f"Sources must be a list, got {type(sources).__name__}."
            )

        warnings: List[str] = []
        valid: List[Dict] = []

        for i, source in enumerate(sources):
            result = cls.validate_source(source)
            if result.ok:
                valid.append(result.data)
                warnings.extend([f"[source {i}] {w}" for w in result.warnings])
            else:
                warnings.append(f"[source {i}] Dropped invalid source: {result.error}")

        return ValidationResult.success(valid, warnings)

    # ------------------------------------------------------------------
    # Research document validation
    # ------------------------------------------------------------------

    @classmethod
    def validate_research_document(cls, doc: Any) -> ValidationResult:
        """
        Validate a research document dict before it is stored in
        research_data["documents"] or retrieved_documents.

        Required: text (non-empty, >= 50 chars)
        Optional: title, url, source_type, relevance_score, metadata
        """
        if not isinstance(doc, dict):
            return ValidationResult.failure(
                f"Document must be a dict, got {type(doc).__name__}."
            )

        warnings: List[str] = []

        text = str(doc.get("text") or doc.get("content") or doc.get("body") or "").strip()
        if not text:
            return ValidationResult.failure("Document 'text' is required and cannot be empty.")
        if len(text) < 50:
            warnings.append(
                f"Document 'text' is very short ({len(text)} chars) — may not be useful for SEO or writing."
            )

        url = str(doc.get("url") or "").strip()
        if url:
            url_result = cls._validate_url(url)
            if not url_result.ok:
                warnings.append(f"Document has invalid URL: {url_result.error}")
                url = ""

        source_type = str(doc.get("source_type") or "web").strip().lower()
        if source_type not in VALID_SOURCE_TYPES:
            warnings.append(f"Unknown source_type '{source_type}' — defaulting to 'web'.")
            source_type = "web"

        score_raw = doc.get("relevance_score", 0.0)
        try:
            relevance_score = float(score_raw)
            relevance_score = max(0.0, min(1.0, relevance_score))
        except (TypeError, ValueError):
            relevance_score = 0.0
            warnings.append("Invalid relevance_score — defaulted to 0.0.")

        clean = {
            "text": text,
            "title": str(doc.get("title") or "").strip(),
            "url": url,
            "source_type": source_type,
            "relevance_score": relevance_score,
            "metadata": doc.get("metadata", {}) if isinstance(doc.get("metadata"), dict) else {},
        }

        return ValidationResult.success(clean, warnings)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_url(url: str) -> "ValidationResult":
        """Check that a URL is well-formed (has scheme + netloc)."""
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                return ValidationResult.failure(
                    f"URL scheme '{parsed.scheme}' is not http/https."
                )
            if not parsed.netloc:
                return ValidationResult.failure("URL has no domain/host.")
            return ValidationResult.success(url)
        except Exception as exc:
            return ValidationResult.failure(f"URL parse error: {exc}")

    @staticmethod
    def _build_formatted(text: str, url: str) -> str:
        """Reconstruct a formatted citation string from available fields."""
        if url:
            return f"{text}. {url}"
        return text

    @staticmethod
    def _deduplicate(citations: List[Dict]) -> tuple[List[Dict], List[str]]:
        """
        Remove duplicate citations by normalised 'formatted' string.
        First occurrence wins. Returns (deduped_list, warnings).
        """
        seen: set = set()
        unique: List[Dict] = []
        warnings: List[str] = []

        for citation in citations:
            key = re.sub(r"\s+", " ", citation.get("formatted", citation.get("text", ""))).strip().lower()[:60]
            if key and key not in seen:
                seen.add(key)
                unique.append(citation)
            elif key:
                warnings.append(f"Duplicate citation removed: '{citation.get('text', '')[:50]}'")

        return unique, warnings
