"""
JSON Validator
==============

Validates and sanitises raw JSON strings returned by LLM calls across
all services (SEOService, ReviewService, HashtagService, CitationService,
WriterService).

Every LLM in this project is instructed to return pure JSON, but models
occasionally wrap it in markdown fences or add prose. This module:

  1. Strips markdown fences and surrounding whitespace.
  2. Parses the JSON safely.
  3. Validates the parsed object against a required shape (keys + types).
  4. Returns a typed ValidationResult so callers never need to try/except.

Usage
-----
    from validators.json_validator import JSONValidator

    result = JSONValidator.validate_keyword_extraction(raw_llm_response)
    if result.ok:
        keywords = result.data          # List[Dict]
    else:
        logger.error(result.error)
        keywords = []
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

logger = logging.getLogger(__name__)

# Valid values for constrained enum fields
VALID_KEYWORD_CATEGORIES = {"primary", "secondary", "long_tail", "industry", "technical"}
VALID_SEARCH_INTENTS = {"Informational", "Commercial", "Transactional", "Navigational"}
VALID_CITATION_TYPES = {"news", "research", "video", "community", "report", "book", "web"}


# ==========================================================================
# Result wrapper
# ==========================================================================


@dataclass
class ValidationResult:
    """Return type for every validator. Never raises — always returns status."""

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
# Core JSON parser
# ==========================================================================


class JSONValidator:
    """
    Static validators for each LLM output shape in the project.
    All methods return ValidationResult — they never raise.
    """

    # ------------------------------------------------------------------
    # Step 0: shared sanitiser (used by every validate_* method)
    # ------------------------------------------------------------------

    @staticmethod
    def sanitise(raw: str) -> str:
        """
        Strip markdown code fences, backticks, and leading/trailing whitespace
        from an LLM response so json.loads() can parse it cleanly.

        Handles:
          ```json ... ```
          ``` ... ```
          ` ... `
          Prose before/after the JSON object or array
        """
        if not raw or not raw.strip():
            return ""

        text = raw.strip()

        # Remove fenced code blocks (```json ... ``` or ``` ... ```)
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = re.sub(r"```", "", text)

        # Remove loose backtick wrappers
        text = text.strip("`").strip()

        # If there is prose before the JSON, extract the first {...} or [...]
        json_match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
        if json_match:
            text = json_match.group(1)

        return text.strip()

    @staticmethod
    def _parse(raw: str) -> tuple[Any, str]:
        """
        Sanitise and parse raw LLM text.
        Returns (parsed_object, error_string). Error is empty string on success.
        """
        cleaned = JSONValidator.sanitise(raw)
        if not cleaned:
            return None, "Empty response after sanitisation."
        try:
            return json.loads(cleaned), ""
        except json.JSONDecodeError as exc:
            snippet = cleaned[:200]
            return None, f"JSON parse error: {exc} | head: {snippet}"

    # ------------------------------------------------------------------
    # Keyword extraction  →  [{"keyword": str, "category": str}]
    # SEOService._extract_candidate_keywords()
    # ------------------------------------------------------------------

    @classmethod
    def validate_keyword_extraction(cls, raw: str) -> ValidationResult:
        """
        Expected: JSON array of {keyword: str, category: str} objects.
        Returns ValidationResult with data=List[Dict] on success.
        """
        parsed, err = cls._parse(raw)
        if err:
            return ValidationResult.failure(f"keyword_extraction: {err}")

        if not isinstance(parsed, list):
            return ValidationResult.failure(
                f"keyword_extraction: expected array, got {type(parsed).__name__}"
            )

        clean: List[Dict] = []
        warnings: List[str] = []

        for i, item in enumerate(parsed):
            if not isinstance(item, dict):
                warnings.append(f"Item {i} is not a dict — skipped.")
                continue

            keyword = str(item.get("keyword", "")).strip().lower()
            category = str(item.get("category", "secondary")).strip().lower()

            if not keyword:
                warnings.append(f"Item {i} has empty keyword — skipped.")
                continue

            if category not in VALID_KEYWORD_CATEGORIES:
                warnings.append(
                    f"Item {i} has unknown category '{category}' — defaulting to 'secondary'."
                )
                category = "secondary"

            clean.append({"keyword": keyword, "category": category})

        if not clean:
            return ValidationResult.failure(
                "keyword_extraction: no valid keyword objects found after parsing."
            )

        return ValidationResult.success(clean, warnings)

    # ------------------------------------------------------------------
    # Intent classification  →  [{"idx": int, "intent": str}]
    # SEOService._classify_intent_batch()
    # ------------------------------------------------------------------

    @classmethod
    def validate_intent_classification(cls, raw: str) -> ValidationResult:
        """
        Expected: JSON array of {idx: int, intent: str} objects.
        Returns ValidationResult with data=Dict[int, str] mapping idx → intent.
        """
        parsed, err = cls._parse(raw)
        if err:
            return ValidationResult.failure(f"intent_classification: {err}")

        if not isinstance(parsed, list):
            return ValidationResult.failure(
                f"intent_classification: expected array, got {type(parsed).__name__}"
            )

        intent_map: Dict[int, str] = {}
        warnings: List[str] = []

        for item in parsed:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item["idx"])
                intent = str(item.get("intent", "Informational")).strip()
                if intent not in VALID_SEARCH_INTENTS:
                    warnings.append(
                        f"Unknown intent '{intent}' for idx {idx} — defaulting to 'Informational'."
                    )
                    intent = "Informational"
                intent_map[idx] = intent
            except (KeyError, ValueError, TypeError) as exc:
                warnings.append(f"Skipping malformed intent item: {exc}")

        return ValidationResult.success(intent_map, warnings)

    # ------------------------------------------------------------------
    # Meta fields  →  {"meta_title": str, "meta_description": str, "slug": str}
    # SEOService._generate_meta_fields()
    # ------------------------------------------------------------------

    @classmethod
    def validate_meta_fields(cls, raw: str) -> ValidationResult:
        """
        Expected: JSON object with meta_title, meta_description, slug.
        Returns ValidationResult with data=Dict[str, str].
        """
        parsed, err = cls._parse(raw)
        if err:
            return ValidationResult.failure(f"meta_fields: {err}")

        if not isinstance(parsed, dict):
            return ValidationResult.failure(
                f"meta_fields: expected object, got {type(parsed).__name__}"
            )

        warnings: List[str] = []
        data: Dict[str, str] = {}

        meta_title = str(parsed.get("meta_title", "")).strip()
        meta_description = str(parsed.get("meta_description", "")).strip()
        slug = str(parsed.get("slug", "")).strip()

        if not meta_title:
            warnings.append("meta_title is empty.")
        elif len(meta_title) > 60:
            meta_title = meta_title[:60]
            warnings.append("meta_title truncated to 60 characters.")

        if not meta_description:
            warnings.append("meta_description is empty.")
        elif len(meta_description) > 160:
            meta_description = meta_description[:160]
            warnings.append("meta_description truncated to 160 characters.")

        if not slug:
            warnings.append("slug is empty.")
        else:
            # Enforce lowercase hyphen-only slug
            slug = re.sub(r"[^a-z0-9-]", "", slug.lower().replace(" ", "-"))
            slug = re.sub(r"-{2,}", "-", slug).strip("-")[:80]

        data = {
            "meta_title": meta_title,
            "meta_description": meta_description,
            "slug": slug,
        }

        return ValidationResult.success(data, warnings)

    # ------------------------------------------------------------------
    # Hashtags  →  {"hashtags": [str]}
    # HashtagService._parse_response()
    # ------------------------------------------------------------------

    @classmethod
    def validate_hashtags(cls, raw: str) -> ValidationResult:
        """
        Expected: JSON object with a "hashtags" key containing a list of strings.
        Returns ValidationResult with data=List[str].
        """
        parsed, err = cls._parse(raw)
        if err:
            return ValidationResult.failure(f"hashtags: {err}")

        if not isinstance(parsed, dict):
            return ValidationResult.failure(
                f"hashtags: expected object, got {type(parsed).__name__}"
            )

        tags_raw = parsed.get("hashtags", [])
        if not isinstance(tags_raw, list):
            return ValidationResult.failure("hashtags: 'hashtags' value is not an array.")

        warnings: List[str] = []
        clean: List[str] = []

        for tag in tags_raw:
            tag = str(tag).strip()
            if not tag:
                continue
            if not tag.startswith("#"):
                tag = "#" + tag
                warnings.append(f"Added missing '#' prefix to tag: {tag}")
            tag = "#" + re.sub(r"\s+", "", tag[1:])
            if len(tag) > 1:
                clean.append(tag)

        if not clean:
            return ValidationResult.failure("hashtags: no valid hashtags found.")

        return ValidationResult.success(clean, warnings)

    # ------------------------------------------------------------------
    # Review evaluation  →  {dimension_scores, feedback, issues, rewrite_instruction}
    # ReviewService._parse_evaluation()
    # ------------------------------------------------------------------

    @classmethod
    def validate_review_evaluation(cls, raw: str) -> ValidationResult:
        """
        Expected: JSON object with dimension_scores dict, feedback list,
        issues list, and rewrite_instruction string.
        Returns ValidationResult with data=Dict.
        """
        parsed, err = cls._parse(raw)
        if err:
            return ValidationResult.failure(f"review_evaluation: {err}")

        if not isinstance(parsed, dict):
            return ValidationResult.failure(
                f"review_evaluation: expected object, got {type(parsed).__name__}"
            )

        warnings: List[str] = []
        required_dims = {
            "content_quality", "seo_compliance",
            "brand_alignment", "structure",
            "factual_grounding", "cta_effectiveness",
        }

        # Validate dimension_scores
        dim_raw = parsed.get("dimension_scores", {})
        if not isinstance(dim_raw, dict):
            return ValidationResult.failure(
                "review_evaluation: 'dimension_scores' must be an object."
            )

        dim_scores: Dict[str, int] = {}
        for dim in required_dims:
            raw_val = dim_raw.get(dim)
            try:
                score = max(0, min(100, int(raw_val if raw_val is not None else 50)))
            except (TypeError, ValueError):
                score = 50
                warnings.append(f"dimension '{dim}' had invalid value — defaulted to 50.")
            dim_scores[dim] = score

        # Validate list fields
        feedback = parsed.get("feedback", [])
        if not isinstance(feedback, list):
            feedback = []
            warnings.append("'feedback' was not an array — defaulted to empty list.")

        issues = parsed.get("issues", [])
        if not isinstance(issues, list):
            issues = []
            warnings.append("'issues' was not an array — defaulted to empty list.")

        rewrite = str(parsed.get("rewrite_instruction", "")).strip()

        data = {
            "dimension_scores": dim_scores,
            "feedback": [str(f).strip() for f in feedback if str(f).strip()],
            "issues": [str(i).strip() for i in issues if str(i).strip()],
            "rewrite_instruction": rewrite,
        }

        return ValidationResult.success(data, warnings)

    # ------------------------------------------------------------------
    # Content outline  →  {"title": str, "content_angle": str, "sections": [...]}
    # WriterService._parse_outline_json()
    # ------------------------------------------------------------------

    @classmethod
    def validate_content_outline(cls, raw: str) -> ValidationResult:
        """
        Expected: JSON object with title, content_angle, and sections array.
        Returns ValidationResult with data=Dict.
        """
        parsed, err = cls._parse(raw)
        if err:
            return ValidationResult.failure(f"content_outline: {err}")

        if not isinstance(parsed, dict):
            return ValidationResult.failure(
                f"content_outline: expected object, got {type(parsed).__name__}"
            )

        warnings: List[str] = []

        title = str(parsed.get("title", "")).strip()
        content_angle = str(parsed.get("content_angle", "")).strip()
        sections_raw = parsed.get("sections", [])

        if not title:
            warnings.append("outline 'title' is empty.")

        if not isinstance(sections_raw, list):
            return ValidationResult.failure("content_outline: 'sections' must be an array.")

        if not sections_raw:
            return ValidationResult.failure("content_outline: 'sections' array is empty.")

        sections: List[Dict] = []
        for i, sec in enumerate(sections_raw):
            if not isinstance(sec, dict):
                warnings.append(f"Section {i} is not a dict — skipped.")
                continue
            heading = str(sec.get("heading") or sec.get("title") or "").strip()
            if not heading:
                warnings.append(f"Section {i} has no heading — skipped.")
                continue
            level = sec.get("heading_level") or sec.get("level") or 2
            try:
                level = max(2, min(3, int(level)))
            except (TypeError, ValueError):
                level = 2
            sections.append({
                "heading": heading,
                "heading_level": level,
                "brief": str(sec.get("brief") or sec.get("description") or "").strip(),
                "keywords": [str(k) for k in sec.get("keywords", []) if k],
            })

        if not sections:
            return ValidationResult.failure(
                "content_outline: no valid sections found after parsing."
            )

        data = {
            "title": title,
            "content_angle": content_angle,
            "sections": sections,
        }
        return ValidationResult.success(data, warnings)

    # ------------------------------------------------------------------
    # Citation enrichment  →  [{"text", "url", "type", "formatted"}]
    # CitationService._enrich_citations_via_llm()
    # ------------------------------------------------------------------

    @classmethod
    def validate_citation_enrichment(cls, raw: str) -> ValidationResult:
        """
        Expected: JSON array of citation objects with text, url, type, formatted.
        Returns ValidationResult with data=List[Dict].
        """
        parsed, err = cls._parse(raw)
        if err:
            return ValidationResult.failure(f"citation_enrichment: {err}")

        if not isinstance(parsed, list):
            return ValidationResult.failure(
                f"citation_enrichment: expected array, got {type(parsed).__name__}"
            )

        warnings: List[str] = []
        clean: List[Dict] = []

        for i, item in enumerate(parsed):
            if not isinstance(item, dict):
                warnings.append(f"Citation {i} is not a dict — skipped.")
                continue

            citation_type = str(item.get("type", "web")).strip().lower()
            if citation_type not in VALID_CITATION_TYPES:
                warnings.append(
                    f"Citation {i} has unknown type '{citation_type}' — defaulting to 'web'."
                )
                citation_type = "web"

            clean.append({
                "text": str(item.get("text", "")).strip(),
                "url": str(item.get("url", "")).strip(),
                "type": citation_type,
                "formatted": str(item.get("formatted", "")).strip(),
            })

        if not clean:
            return ValidationResult.failure(
                "citation_enrichment: no valid citation objects found."
            )

        return ValidationResult.success(clean, warnings)
