"""
API Schemas
===========

Pydantic v2 request and response models for all FastAPI endpoints.

Request models validate incoming JSON payloads.
Response models document the API contract and serialize workflow results.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ==========================================================================
# Request Models
# ==========================================================================


class ContentRequest(BaseModel):
    """POST /api/generate/content"""

    user_input: str = Field(..., min_length=3, description="Topic, question, or content brief.")
    content_type: str = Field("article", description="article | blog")
    brand: Optional[str] = Field(None, description="Brand name or alias (e.g. 'Futuristix').")
    objective: str = Field("seo", description="seo | authority | engagement | leads")
    content_mode: Optional[str] = Field(
        None,
        description="Optional override: awareness | authority | lead_gen | seo_page",
    )
    language: str = Field("English", description="English | Hindi")
    additional_instructions: str = Field("", description="Extra writer guidance.")
    session_id: Optional[str] = Field(None, description="Session ID for ConversationMemory.")
    max_revisions: int = Field(
        2, ge=1, le=5, description="Max review→writer cycles (default 2 toward 95+)."
    )

    model_config = {"json_schema_extra": {
        "example": {
            "user_input": "How AI agents are transforming SMB operations",
            "content_type": "article",
            "brand": "Futuristix",
            "objective": "seo",
            "language": "English",
        }
    }}


class EmailRequest(BaseModel):
    """POST /api/generate/email"""

    user_input: str = Field(..., min_length=3, description="Topic, offer, or email brief.")
    brand: Optional[str] = Field(None, description="Brand name or alias.")
    campaign_type: str = Field("newsletter", description="newsletter | nurture | promotional | transactional")
    objective: str = Field("leads", description="leads | engagement")
    content_mode: Optional[str] = Field(
        None, description="Optional: awareness | authority | lead_gen | seo_page"
    )
    language: str = Field("English", description="English | Hindi")
    additional_instructions: str = Field("", description="Extra writer guidance.")
    session_id: Optional[str] = Field(None, description="Session ID for ConversationMemory.")
    max_revisions: int = Field(1, ge=1, le=4, description="Max review→writer cycles.")

    model_config = {"json_schema_extra": {
        "example": {
            "user_input": "Announce our new AI audit service to founders",
            "brand": "Futuristix",
            "campaign_type": "promotional",
            "objective": "leads",
        }
    }}


class SEORequest(BaseModel):
    """POST /api/generate/seo"""

    user_input: str = Field(..., min_length=3, description="Search query or content brief.")
    content_type: str = Field("article", description="article | blog")
    brand: Optional[str] = Field(None, description="Brand name or alias.")
    content_mode: Optional[str] = Field(
        None, description="Optional: awareness | authority | lead_gen | seo_page"
    )
    language: str = Field("English", description="English | Hindi")
    additional_instructions: str = Field("", description="Extra writer guidance.")
    session_id: Optional[str] = Field(None, description="Session ID for ConversationMemory.")
    max_revisions: int = Field(
        2, ge=1, le=5, description="Max review→writer cycles (default 2 toward 95+)."
    )

    model_config = {"json_schema_extra": {
        "example": {
            "user_input": "AI agents for small business automation",
            "content_type": "article",
            "brand": "Futuristix",
        }
    }}


class SocialRequest(BaseModel):
    """POST /api/generate/social"""

    user_input: str = Field(..., min_length=3, description="Topic or brief for the post.")
    platform: str = Field(
        "linkedin",
        description="linkedin | carousel | x | instagram | facebook | reddit | comment",
    )
    brand: Optional[str] = Field(None, description="Brand name or alias.")
    objective: str = Field("engagement", description="engagement | authority | leads")
    content_mode: Optional[str] = Field(
        None, description="Optional: awareness | authority | lead_gen | seo_page"
    )
    language: str = Field("English", description="English | Hindi")
    additional_instructions: str = Field("", description="Extra writer guidance.")
    session_id: Optional[str] = Field(None, description="Session ID for ConversationMemory.")
    max_revisions: int = Field(1, ge=1, le=4, description="Max review→writer cycles.")

    model_config = {"json_schema_extra": {
        "example": {
            "user_input": "Why AI agents are the next competitive advantage for SMBs",
            "platform": "linkedin",
            "brand": "Futuristix",
            "objective": "engagement",
        }
    }}


# ==========================================================================
# Response Models
# ==========================================================================


class ReviewSummary(BaseModel):
    score: int = 0
    status: str = ""
    needs_revision: bool = False
    feedback: List[str] = []
    issues: List[str] = []
    dimension_scores: Dict[str, Any] = {}
    below_target: bool = False
    quality_label: str = ""
    final_qc: Dict[str, Any] = {}

class GenerateRequest(BaseModel):
    user_input: str = Field(..., min_length=3)
    brand: Optional[str] = None
    language: str = "English"
    additional_instructions: str = ""
    session_id: Optional[str] = None
    max_revisions: int = Field(2, ge=1, le=5)


class WorkflowResult(BaseModel):
    """Base response returned by all workflow endpoints."""

    ok: bool
    request_id: str
    session_id: str
    workflow_status: str
    review: ReviewSummary = ReviewSummary()
    revision_count: int = 0
    metadata: Dict[str, Any] = {}
    final_output: Dict[str, Any] = {}
    errors: List[str] = []
    safety: Dict[str, Any] = {}
    primary_topic: str = ""
    user_constraints: Dict[str, Any] = {}
    content_mode: str = ""
    mode_policy: Dict[str, Any] = {}

    model_config = {"arbitrary_types_allowed": True}


class ContentResult(WorkflowResult):
    """Response from POST /api/generate/content"""
    pass

class GenerateResult(WorkflowResult):
    pass


class EmailResult(WorkflowResult):
    """Response from POST /api/generate/email"""
    email_meta: Dict[str, Any] = {}


class SEOResult(WorkflowResult):
    """Response from POST /api/generate/seo"""
    seo_analysis: Dict[str, Any] = {}


class SocialResult(WorkflowResult):
    """Response from POST /api/generate/social"""
    social_meta: Dict[str, Any] = {}


# ==========================================================================
# Utility Response Models
# ==========================================================================


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    environment: str


class BrandInfo(BaseModel):
    id: str
    display_name: str
    tone: str
    reader_segment: List[str]
    cta: str
    namespace: str
    font: str = ""


class BrandsResponse(BaseModel):
    brands: List[BrandInfo]
    total: int


class ErrorResponse(BaseModel):
    ok: bool = False
    error: str
    detail: Optional[str] = None


class SurgicalEditRequest(BaseModel):
    """POST /api/editor/surgical — fix one Final QC flag family."""

    draft: str = Field(..., min_length=20)
    flags: List[str] = Field(
        ...,
        description="Flag text or keys: seo_stuff, duplicate_cta, thesis_repeat, mode_h1, …",
    )
    brand: Optional[str] = None
    content_mode: Optional[str] = None
    primary_topic: str = ""
    cta: str = ""
    content_type: str = "article"
    strategy: Dict[str, Any] = Field(default_factory=dict)
    brand_context: Dict[str, Any] = Field(default_factory=dict)


class SurgicalEditResult(BaseModel):
    ok: bool = True
    draft: str = ""
    final_qc: Dict[str, Any] = Field(default_factory=dict)


class ApproveGoldRequest(BaseModel):
    """POST /api/editor/approve-gold — save draft into evals/gold."""

    draft: str = Field(..., min_length=20)
    user_input: str = ""
    brand: Optional[str] = None
    content_mode: str = "seo_page"
    primary_topic: str = ""
    must_include: List[str] = Field(default_factory=list)
    must_not: List[str] = Field(default_factory=list)
    fixture_id: Optional[str] = None
    save_reference: bool = True


class ApproveGoldResult(BaseModel):
    ok: bool = True
    fixture_id: str = ""
    path: str = ""
