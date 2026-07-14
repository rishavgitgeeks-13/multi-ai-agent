"""
Unified generation endpoint.

Automatically routes requests to the correct workflow:

- Content
- Email
- Social
- SEO

Workflow selection and brand selection are handled by
BusinessContextService.
"""

import asyncio
import logging

from fastapi import APIRouter, HTTPException

from api.routes import _normalise
from api.schemas import (
    GenerateRequest,
    GenerateResult,
)
from services.business_context_service import (
    BusinessContextService,
)
from workflows.content_workflow import ContentWorkflow
from workflows.email_workflow import EmailWorkflow
from workflows.seo_workflow import SEOWorkflow
from workflows.social_workflow import SocialWorkflow

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Generate"])

# ------------------------------------------------------------------
# Shared instances
# ------------------------------------------------------------------

_business_context = BusinessContextService()

_content_workflow = ContentWorkflow()
_email_workflow = EmailWorkflow()
_seo_workflow = SEOWorkflow()
_social_workflow = SocialWorkflow()


@router.post(
    "/generate",
    response_model=GenerateResult,
    summary="Automatically generate content using the appropriate workflow",
    description=(
        "Automatically detects whether the request is for "
        "content, email, social, or SEO generation and "
        "executes the appropriate workflow."
    ),
)
async def generate(
    req: GenerateRequest,
) -> GenerateResult:

    logger.info(
        "POST /generate | user_input=%s",
        req.user_input[:100],
    )

    try:
        context = _business_context.resolve(
            user_input=req.user_input,
            brand=req.brand,
        )

        workflow = context["workflow"]
        brand_cfg = context["brand_config"]

        brand_name = (
            brand_cfg.get("namespace")
            or brand_cfg.get("display_name")
            or req.brand
        )

        logger.info(
            "Auto-routed | workflow=%s | brand=%s",
            workflow,
            brand_name,
        )

        # --------------------------------------------------
        # Content
        # --------------------------------------------------
        if workflow == "content":

            result = await asyncio.to_thread(
                lambda: _content_workflow.run(
                    user_input=req.user_input,
                    content_type=context["content_type"],
                    brand=brand_name,
                    objective=context["objective"],
                    language=req.language,
                    additional_instructions=req.additional_instructions,
                    session_id=req.session_id,
                    max_revisions=req.max_revisions,
                )
            )

        # --------------------------------------------------
        # Email
        # --------------------------------------------------
        elif workflow == "email":

            result = await asyncio.to_thread(
                lambda: _email_workflow.run(
                    user_input=req.user_input,
                    campaign_type=context["campaign_type"],
                    brand=brand_name,
                    objective=context["objective"],
                    language=req.language,
                    additional_instructions=req.additional_instructions,
                    session_id=req.session_id,
                    max_revisions=req.max_revisions,
                )
            )

        # --------------------------------------------------
        # Social
        # --------------------------------------------------
        elif workflow == "social":

            result = await asyncio.to_thread(
                lambda: _social_workflow.run(
                    user_input=req.user_input,
                    platform=context["platform"],
                    brand=brand_name,
                    objective=context["objective"],
                    language=req.language,
                    additional_instructions=req.additional_instructions,
                    session_id=req.session_id,
                    max_revisions=req.max_revisions,
                )
            )

        # --------------------------------------------------
        # SEO
        # --------------------------------------------------
        elif workflow == "seo":

            result = await asyncio.to_thread(
                lambda: _seo_workflow.run(
                    user_input=req.user_input,
                    content_type=context["content_type"],
                    brand=brand_name,
                    language=req.language,
                    additional_instructions=req.additional_instructions,
                    session_id=req.session_id,
                    max_revisions=req.max_revisions,
                )
            )

        else:
            raise ValueError(
                f"Unsupported workflow '{workflow}'"
            )

        logger.info(
            "Workflow completed | workflow=%s | status=%s",
            workflow,
            result.get(
                "workflow_status",
                "unknown",
            ),
        )

        return GenerateResult(
            **_normalise(result)
        )

    except Exception as exc:
        logger.error(
            "Unified generation endpoint failed: %s",
            exc,
            exc_info=True,
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )