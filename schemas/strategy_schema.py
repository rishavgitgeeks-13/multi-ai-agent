"""
Strategy Schema
===============

Pydantic models for the data contract produced by the Strategy Agent
and consumed by the Writer Agent.

Covers:
  - KeywordScore   — per-keyword scoring breakdown from SEOService
  - SEOBlueprint   — full SEO output (primary/secondary keywords, meta, slug)
  - OutlineSection — one section in the content plan
  - Strategy       — complete strategy dict stored in ContentState["strategy"]
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ==========================================================================
# SEO sub-models
# ==========================================================================


class KeywordScore(BaseModel):
    """
    Scoring breakdown for one keyword candidate produced by SEOService.
    Stored in SEOBlueprint.keyword_scores and exposed in ContentState["seo"].
    """

    keyword: str
    category: str
    """primary | secondary | long_tail | industry | technical"""
    search_intent: str = ""
    """Informational | Commercial | Transactional | Navigational"""
    scores: Dict[str, float] = Field(default_factory=dict)
    """
    Dimension scores (all in [0, 1]):
      semantic_similarity  — keyword vs. user query (OpenAI embeddings)
      tfidf                — TF-IDF cosine vs. corpus centroid
      bm25                 — BM25Okapi best-match score (normalized)
      pain_point           — keyword vs. brand pain points (embeddings)
      brand_relevance      — keyword vs. keyword direction (embeddings)
      search_intent        — numeric mapping of the intent label
    """
    final_score: float = Field(default=0.0, ge=0.0, le=1.0)


class SEOBlueprint(BaseModel):
    """
    Full SEO output from SEOService.
    Stored in both ContentState["seo"] and ContentState["strategy"]["seo"].
    """

    primary_keywords: List[str] = Field(default_factory=list)
    """Top-5 keywords ranked by final_score — lead with these in H1/H2."""
    secondary_keywords: List[str] = Field(default_factory=list)
    """Secondary and long-tail keywords — distribute through the body."""
    keyword_scores: List[KeywordScore] = Field(default_factory=list)
    """Full ranked breakdown for audit / analytics use."""
    search_intent: str = "Informational"
    """Dominant intent across the top-10 ranked keywords."""
    meta_title: str = ""
    """50–60 chars, primary keyword first."""
    meta_description: str = ""
    """150–160 chars, benefit-driven, ends with a CTA."""
    slug: str = ""
    """Lowercase, hyphen-separated, 3–6 words."""

    @field_validator("meta_title")
    @classmethod
    def _clamp_meta_title(cls, v: str) -> str:
        return v[:60]

    @field_validator("meta_description")
    @classmethod
    def _clamp_meta_description(cls, v: str) -> str:
        return v[:160]

    @field_validator("slug")
    @classmethod
    def _clamp_slug(cls, v: str) -> str:
        return v[:80]


# ==========================================================================
# Outline sub-model
# ==========================================================================


class OutlineSection(BaseModel):
    """One section in the content plan, consumed by WriterService."""

    heading: str
    heading_level: int = Field(default=2, ge=2, le=3)
    """2 = H2, 3 = H3"""
    brief: str = ""
    """1–2 sentences describing what this section must cover."""
    keywords: List[str] = Field(default_factory=list)
    """Target keywords to weave naturally into this section."""


# ==========================================================================
# Top-level Strategy contract
# ==========================================================================


class Strategy(BaseModel):
    """
    Complete strategy dict produced by the Strategy Agent and stored in
    ContentState["strategy"].

    The Writer Agent reads every field here when generating the draft.
    """

    # ------------------------------------------------------------------
    # Core content plan
    # ------------------------------------------------------------------
    title: str = ""
    """Primary H1 title — keyword-rich and benefit-driven."""
    content_angle: str = ""
    """Unique hook or narrative angle that differentiates this piece."""
    audience: List[str] = Field(default_factory=list)
    """Target reader segments (e.g. ['B2B SaaS founders', 'CMOs'])."""
    tone: str = ""
    """Brand tone applied throughout (e.g. 'ROI-driven', 'Conversational')."""
    outline: List[OutlineSection] = Field(default_factory=list)
    """Ordered section plan (problem → solution → proof → CTA)."""
    cta: str = ""
    """Call-to-action text inserted at the end of the piece."""

    # ------------------------------------------------------------------
    # Request metadata (passed through from ContentState)
    # ------------------------------------------------------------------
    content_type: str = "article"
    """article | blog | linkedin | email | carousel"""
    platform: str = "website"
    """website | linkedin | email | x"""
    language: str = "English"

    # ------------------------------------------------------------------
    # Keyword strategy (from SEOService)
    # ------------------------------------------------------------------
    keywords: List[str] = Field(default_factory=list)
    """Primary keywords — top-5 from SEOBlueprint."""
    secondary_keywords: List[str] = Field(default_factory=list)
    """Secondary / long-tail keywords."""
    pain_points: List[str] = Field(default_factory=list)
    """Brand pain points to address in the content."""

    # ------------------------------------------------------------------
    # Full SEO blueprint
    # ------------------------------------------------------------------
    seo: SEOBlueprint = Field(default_factory=SEOBlueprint)
    """Complete SEOBlueprint — read by Formatter and JSONBuilder."""

    # ------------------------------------------------------------------
    # Hashtags and citations
    # ------------------------------------------------------------------
    hashtags: List[str] = Field(default_factory=list)
    """Platform-optimised hashtags from HashtagService."""
    citations: List[str] = Field(default_factory=list)
    """Formatted citations from CitationService."""

    # ------------------------------------------------------------------
    # Revision control (injected by Review Agent on FAIL)
    # ------------------------------------------------------------------
    rewrite_instruction: str = ""
    """
    Actionable revision brief from ReviewService.
    Injected into WriterService prompts on the next rewrite pass.
    Empty string when the content passed review.
    """

    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize to the dict stored in ContentState['strategy']."""
        data = self.model_dump()
        # Flatten keyword_scores inside seo for JSON compatibility
        data["seo"]["keyword_scores"] = [
            ks.model_dump() if isinstance(ks, KeywordScore) else ks
            for ks in (self.seo.keyword_scores or [])
        ]
        return data

    @classmethod
    def from_state_dict(cls, data: Dict[str, Any]) -> "Strategy":
        """Reconstruct from ContentState['strategy']."""
        return cls(**data)
