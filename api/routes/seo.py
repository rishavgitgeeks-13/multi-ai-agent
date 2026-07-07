"""SEO content generation endpoint with full keyword analysis."""

import asyncio
import logging

from fastapi import APIRouter, HTTPException

from api.schemas import SEORequest, SEOResult
from api.routes import _normalise
from workflows.seo_workflow import SEOWorkflow

logger = logging.getLogger(__name__)

router = APIRouter(tags=["SEO"])

_workflow = SEOWorkflow()


@router.post(
    "/generate/seo",
    response_model=SEOResult,
    summary="Generate SEO-optimised article with keyword analysis",
    description=(
        "Runs the 5-agent pipeline with objective='seo' and injects explicit SEO "
        "writer instructions. Returns the content plus a detailed seo_analysis block: "
        "keyword density, heading audit, technical SEO checklist, and per-keyword scores."
    ),
)
async def generate_seo(req: SEORequest) -> SEOResult:
    logger.info(
        "POST /generate/seo | content_type=%s | brand=%s",
        req.content_type, req.brand,
    )
    try:
        result = await asyncio.to_thread(
            lambda: _workflow.run(
                user_input=req.user_input,
                content_type=req.content_type,
                brand=req.brand,
                language=req.language,
                additional_instructions=req.additional_instructions,
                session_id=req.session_id,
                max_revisions=req.max_revisions,
            )
        )
    except Exception as exc:
        logger.error("SEOWorkflow unhandled error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    return SEOResult(**_normalise(result))
