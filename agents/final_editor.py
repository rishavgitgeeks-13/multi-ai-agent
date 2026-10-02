"""
Final Editor Agent
==================

Runs after Review when the draft is not sent back for a full rewrite
(PASS or force-PASS). Applies surgical QC edits, then completes the workflow.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from schemas.state import ContentState
from services.final_editor_service import FinalEditorService
from services.formatter import Formatter
from services.json_builder import JSONBuilder
from services.metadata_service import MetadataService

logger = logging.getLogger(__name__)

_editor = FinalEditorService()
_formatter = Formatter()
_metadata = MetadataService()
_json_builder = JSONBuilder()


def final_editor_node(state: ContentState) -> Dict[str, Any]:
    """Surgical edit pass → rebuild final_output → COMPLETED."""
    draft = state.get("draft") or ""
    strategy = state.get("strategy") or {}
    brand_context = state.get("brand_context") or {}
    review = dict(state.get("review") or {})
    primary_topic = state.get("primary_topic") or state.get("user_input") or ""

    if state.get("workflow_status") == "BLOCKED" or not draft.strip():
        return {
            "current_agent": "final_editor",
            "next_agent": "end",
            "workflow_status": state.get("workflow_status") or "COMPLETED",
        }

    result = _editor.run(
        draft=draft,
        strategy=strategy,
        brand_context=brand_context,
        primary_topic=str(primary_topic),
        review=review,
    )
    new_draft = result.get("draft") or draft
    final_qc = result.get("final_qc") or {}

    issues = list(review.get("issues") or [])
    for flag in final_qc.get("flags_before") or final_qc.get("flags") or []:
        if flag not in issues:
            issues.append(flag)
    feedback = list(review.get("feedback") or [])
    if final_qc.get("edited"):
        feedback.append(
            "Final Editor applied surgical fixes "
            f"(QC flags remaining after edit: {final_qc.get('flag_count', 0)})."
        )
    elif final_qc.get("flag_count"):
        feedback.append(
            f"Final QC noted {final_qc.get('flag_count')} editorial flags "
            "(no material rewrite needed or edit skipped)."
        )

    if final_qc.get("below_target"):
        score = final_qc.get("review_score") or review.get("score") or 0
        note = (
            f"BELOW TARGET: completed at score {score}/100 "
            "(quality target is 95+). Treat as provisional — not a 9+ ship."
        )
        if note not in feedback:
            feedback.append(note)
        review["below_target"] = True
        review["quality_label"] = "below_target"
    else:
        review["below_target"] = bool(review.get("below_target"))
        if int(review.get("score") or 0) >= 95:
            review["quality_label"] = "on_target"
        else:
            review["quality_label"] = review.get("quality_label") or "reviewed"

    review["issues"] = issues
    review["feedback"] = feedback
    review["final_qc"] = final_qc

    out: Dict[str, Any] = {
        "draft": new_draft,
        "review": review,
        "current_agent": "final_editor",
        "next_agent": "end",
        "workflow_status": "COMPLETED",
    }
    try:
        metadata = _metadata.run(draft=new_draft, strategy=strategy)
        formatted = _formatter.run(draft=new_draft, strategy=strategy)
        final_output = _json_builder.run(
            content=formatted,
            metadata=metadata,
            strategy=strategy,
        )
        out["metadata"] = metadata
        out["formatted_output"] = formatted
        out["final_output"] = final_output
    except Exception as exc:
        logger.warning("Final editor packaging rebuild failed: %s", exc)
        fo = dict(state.get("final_output") or {})
        content = dict(fo.get("content") or {})
        if content or fo:
            content["markdown"] = new_draft
            fo["content"] = content
            out["final_output"] = fo

    logger.info(
        "final_editor_node | edited=%s | qc_flags=%s | below_target=%s | score=%s",
        final_qc.get("edited"),
        final_qc.get("flag_count"),
        final_qc.get("below_target"),
        review.get("score"),
    )
    return out
