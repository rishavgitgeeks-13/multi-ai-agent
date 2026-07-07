"""Email campaign generation endpoint."""

import asyncio
import logging

from fastapi import APIRouter, HTTPException

from api.schemas import EmailRequest, EmailResult
from api.routes import _normalise
from workflows.email_workflow import EmailWorkflow

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Email"])

_workflow = EmailWorkflow()


@router.post(
    "/generate/email",
    response_model=EmailResult,
    summary="Generate email campaign content",
    description=(
        "Runs the 5-agent pipeline tuned for email: short-form (~400 words), "
        "persuasion-driven, with automatic subject line and preview text extraction. "
        "Supports newsletter, nurture, promotional, and transactional campaign types."
    ),
)
async def generate_email(req: EmailRequest) -> EmailResult:
    logger.info(
        "POST /generate/email | campaign_type=%s | brand=%s",
        req.campaign_type, req.brand,
    )
    try:
        result = await asyncio.to_thread(
            lambda: _workflow.run(
                user_input=req.user_input,
                brand=req.brand,
                campaign_type=req.campaign_type,
                objective=req.objective,
                language=req.language,
                additional_instructions=req.additional_instructions,
                session_id=req.session_id,
                max_revisions=req.max_revisions,
            )
        )
    except Exception as exc:
        logger.error("EmailWorkflow unhandled error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    return EmailResult(**_normalise(result))
