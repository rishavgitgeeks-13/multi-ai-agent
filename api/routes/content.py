"""Content generation endpoint — article and blog."""

import asyncio
import logging

from fastapi import APIRouter, HTTPException

from api.schemas import ContentRequest, ContentResult
from api.routes import _normalise
from workflows.content_workflow import ContentWorkflow

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Content"])

# One workflow instance shared across all requests (stateless run() method)
_workflow = ContentWorkflow()


@router.post(
    "/generate/content",
    response_model=ContentResult,
    summary="Generate article or blog content",
    description=(
        "Runs the full 5-agent pipeline (Manager → Research → Strategy → Writer → Review) "
        "optimised for long-form content. Supports article (~2200 words) and blog (~1800 words)."
    ),
)
async def generate_content(req: ContentRequest) -> ContentResult:
    logger.info(
        "POST /generate/content | content_type=%s | brand=%s",
        req.content_type, req.brand,
    )
    try:
        result = await asyncio.to_thread(
            lambda: _workflow.run(
                user_input=req.user_input,
                content_type=req.content_type,
                brand=req.brand,
                objective=req.objective,
                language=req.language,
                additional_instructions=req.additional_instructions,
                session_id=req.session_id,
                max_revisions=req.max_revisions,
            )
        )
    except Exception as exc:
        logger.error("ContentWorkflow unhandled error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    return ContentResult(**_normalise(result))
