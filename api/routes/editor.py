"""Editor UX endpoints — surgical Final QC fix + approve-as-gold."""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException

from api.schemas import (
    ApproveGoldRequest,
    ApproveGoldResult,
    SurgicalEditRequest,
    SurgicalEditResult,
)
from config.mode_policies import attach_mode_to_brand_context, get_policy
from services.final_editor_service import FinalEditorService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Editor"])

_GOLD_ROOT = Path(__file__).resolve().parents[2] / "evals" / "gold"
_editor = FinalEditorService()


@router.post(
    "/editor/surgical",
    response_model=SurgicalEditResult,
    summary="Surgical Final QC fix for one flag family",
)
async def surgical_edit(req: SurgicalEditRequest) -> SurgicalEditResult:
    try:
        brand_context = dict(req.brand_context or {})
        if req.brand and not brand_context.get("display_name"):
            brand_context["display_name"] = req.brand
        if req.cta:
            brand_context["cta"] = req.cta
        brand_context = attach_mode_to_brand_context(
            brand_context,
            content_mode=req.content_mode or brand_context.get("content_mode") or "",
            user_input=req.primary_topic or "",
            primary_topic=req.primary_topic or "",
        )
        strategy = dict(req.strategy or {})
        strategy.setdefault("content_type", req.content_type)
        if req.cta:
            strategy.setdefault("cta", req.cta)
        strategy["mode_policy"] = brand_context.get("mode_policy") or {}

        result = _editor.run(
            draft=req.draft,
            strategy=strategy,
            brand_context=brand_context,
            primary_topic=req.primary_topic,
            only_flags=list(req.flags or []),
        )
        return SurgicalEditResult(
            ok=True,
            draft=result.get("draft") or req.draft,
            final_qc=result.get("final_qc") or {},
        )
    except Exception as exc:
        logger.error("surgical_edit failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post(
    "/editor/approve-gold",
    response_model=ApproveGoldResult,
    summary="Approve draft as a gold-set fixture",
)
async def approve_gold(req: ApproveGoldRequest) -> ApproveGoldResult:
    try:
        _GOLD_ROOT.mkdir(parents=True, exist_ok=True)
        slug = (req.fixture_id or "").strip()
        if not slug:
            base = re.sub(
                r"[^a-z0-9]+",
                "_",
                (req.primary_topic or req.user_input or "gold")[:48].lower(),
            ).strip("_")
            slug = f"{base}_{uuid.uuid4().hex[:6]}"
        fixture_dir = _GOLD_ROOT / slug
        fixture_dir.mkdir(parents=True, exist_ok=True)

        mode = (req.content_mode or "seo_page").strip().lower()
        policy = get_policy(mode)
        brief = {
            "user_input": req.user_input or req.primary_topic,
            "brand": req.brand,
            "expected_mode": mode,
            "mode_policy": policy,
            "primary_topic": req.primary_topic,
            "must_include": list(req.must_include or []),
            "must_not": list(req.must_not or []),
            "approved_at": datetime.now(timezone.utc).isoformat(),
        }
        (fixture_dir / "brief.json").write_text(
            json.dumps(brief, indent=2), encoding="utf-8"
        )
        if req.save_reference:
            (fixture_dir / "reference.md").write_text(req.draft, encoding="utf-8")

        return ApproveGoldResult(
            ok=True,
            fixture_id=slug,
            path=str(fixture_dir),
        )
    except Exception as exc:
        logger.error("approve_gold failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
