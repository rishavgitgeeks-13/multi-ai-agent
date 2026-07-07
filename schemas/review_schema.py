"""
Review Schema
=============

Pydantic models for the data contract produced by the Review Agent
and consumed by the routing logic and workflow tracking.

ReviewService.run() must return a dict valid against ReviewResult.
"""

from typing import Dict, List, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# PASS threshold — must match ReviewService.PASS_THRESHOLD
PASS_THRESHOLD = 70

# Dimension weights — must match ReviewService.DIMENSION_WEIGHTS
DIMENSION_WEIGHTS: Dict[str, float] = {
    "content_quality": 0.25,
    "seo_compliance": 0.25,
    "brand_alignment": 0.20,
    "structure": 0.20,
    "cta_effectiveness": 0.10,
}


# ==========================================================================
# Sub-models
# ==========================================================================


class DimensionScores(BaseModel):
    """
    Per-dimension scores (0–100) from the LLM evaluation.

    Weights applied to produce the final composite score:
      content_quality   25%
      seo_compliance    25%
      brand_alignment   20%
      structure         20%
      cta_effectiveness 10%
    """

    content_quality: int = Field(default=0, ge=0, le=100)
    """Depth, clarity, value, and factual grounding."""
    seo_compliance: int = Field(default=0, ge=0, le=100)
    """Keyword density, heading coverage, meta field quality."""
    brand_alignment: int = Field(default=0, ge=0, le=100)
    """Tone match, audience fit, pain points addressed."""
    structure: int = Field(default=0, ge=0, le=100)
    """Intro / body / conclusion flow and heading hierarchy."""
    cta_effectiveness: int = Field(default=0, ge=0, le=100)
    """CTA clarity, specificity, and intent alignment."""

    def weighted_score(self) -> int:
        """Compute the weighted composite score (rounds to nearest integer)."""
        raw = sum(
            getattr(self, dim) * weight
            for dim, weight in DIMENSION_WEIGHTS.items()
        )
        return round(raw)

    def to_dict(self) -> Dict[str, int]:
        return self.model_dump()


# ==========================================================================
# Top-level Review Result contract
# ==========================================================================


class ReviewResult(BaseModel):
    """
    Full review decision produced by ReviewService and stored in
    ContentState["review"].

    The Review Agent reads this to decide: PASS → END, FAIL → writer.
    The routing function (graphs/routing.py) reads `needs_revision`.
    """

    score: int = Field(default=0, ge=0, le=100)
    """Weighted composite score (0–100). PASS if >= 70."""
    status: Literal["PASS", "FAIL"] = "FAIL"
    needs_revision: bool = True

    feedback: List[str] = Field(default_factory=list)
    """Specific positive observations — what the content does well."""
    issues: List[str] = Field(default_factory=list)
    """
    Problems found — combines rule-based pre-checks
    (word count, keyword presence, heading structure, CTA)
    with LLM-identified issues.
    """
    rewrite_instruction: str = ""
    """
    Actionable revision brief for the Writer Agent.
    Injected into ContentState["strategy"]["rewrite_instruction"] on FAIL.
    Empty string when needs_revision is False.
    """
    dimension_scores: DimensionScores = Field(default_factory=DimensionScores)
    revision_number: int = Field(default=1, ge=1)
    """1-indexed revision counter (revision_count + 1 at time of review)."""

    @model_validator(mode="after")
    def _sync_status_and_flag(self) -> "ReviewResult":
        """Keep status and needs_revision consistent with the score."""
        self.status = "PASS" if self.score >= PASS_THRESHOLD else "FAIL"
        self.needs_revision = self.status == "FAIL"
        if not self.needs_revision:
            self.rewrite_instruction = ""
        return self

    @field_validator("score")
    @classmethod
    def _clamp_score(cls, v: int) -> int:
        return max(0, min(100, v))

    def to_state_dict(self) -> dict:
        """Serialize to the dict stored in ContentState['review']."""
        data = self.model_dump()
        data["dimension_scores"] = self.dimension_scores.to_dict()
        return data

    @classmethod
    def from_service_dict(cls, data: dict) -> "ReviewResult":
        """
        Reconstruct from the raw dict returned by ReviewService.run().
        Handles both flat and nested dimension_scores formats.
        """
        dim_raw = data.get("dimension_scores", {})
        if isinstance(dim_raw, dict):
            data["dimension_scores"] = DimensionScores(**dim_raw)
        return cls(**data)
