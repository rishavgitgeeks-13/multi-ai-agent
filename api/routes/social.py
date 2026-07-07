"""Social media content generation endpoint."""

import asyncio
import logging

from fastapi import APIRouter, HTTPException

from api.schemas import SocialRequest, SocialResult
from api.routes import _normalise
from workflows.social_workflow import SocialWorkflow

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Social"])

_workflow = SocialWorkflow()


@router.post(
    "/generate/social",
    response_model=SocialResult,
    summary="Generate social media content",
    description=(
        "Runs the 5-agent pipeline tuned for social media. "
        "Supports LinkedIn posts (~600 words), carousels (6–10 slides), "
        "and X threads (280-char tweet format). "
        "Returns content plus social_meta: engagement hook, hashtags, character count, slide count."
    ),
)
async def generate_social(req: SocialRequest) -> SocialResult:
    logger.info(
        "POST /generate/social | platform=%s | brand=%s",
        req.platform, req.brand,
    )
    try:
        result = await asyncio.to_thread(
            lambda: _workflow.run(
                user_input=req.user_input,
                platform=req.platform,
                brand=req.brand,
                objective=req.objective,
                language=req.language,
                additional_instructions=req.additional_instructions,
                session_id=req.session_id,
                max_revisions=req.max_revisions,
            )
        )
    except Exception as exc:
        logger.error("SocialWorkflow unhandled error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    return SocialResult(**_normalise(result))
