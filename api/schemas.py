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
    language: str = Field("English", description="English | Hindi")
    additional_instructions: str = Field("", description="Extra writer guidance.")
    session_id: Optional[str] = Field(None, description="Session ID for ConversationMemory.")
    max_revisions: int = Field(3, ge=1, le=5, description="Max review→writer cycles.")

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
    language: str = Field("English", description="English | Hindi")
    additional_instructions: str = Field("", description="Extra writer guidance.")
    session_id: Optional[str] = Field(None, description="Session ID for ConversationMemory.")
    max_revisions: int = Field(2, ge=1, le=4, description="Max review→writer cycles.")

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
    language: str = Field("English", description="English | Hindi")
    additional_instructions: str = Field("", description="Extra writer guidance.")
    session_id: Optional[str] = Field(None, description="Session ID for ConversationMemory.")
    max_revisions: int = Field(3, ge=1, le=5, description="Max review→writer cycles.")

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
    platform: str = Field("linkedin", description="linkedin | carousel | x")
    brand: Optional[str] = Field(None, description="Brand name or alias.")
    objective: str = Field("engagement", description="engagement | authority | leads")
    language: str = Field("English", description="English | Hindi")
    additional_instructions: str = Field("", description="Extra writer guidance.")
    session_id: Optional[str] = Field(None, description="Session ID for ConversationMemory.")
    max_revisions: int = Field(2, ge=1, le=4, description="Max review→writer cycles.")

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

    model_config = {"arbitrary_types_allowed": True}


class ContentResult(WorkflowResult):
    """Response from POST /api/generate/content"""
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


class BrandsResponse(BaseModel):
    brands: List[BrandInfo]
    total: int


class ErrorResponse(BaseModel):
    ok: bool = False
    error: str
    detail: Optional[str] = None
