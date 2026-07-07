"""
Shared utility for normalising workflow result dicts before building
Pydantic response models.
"""

from typing import Any, Dict


def _normalise(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure the workflow result has all fields required by WorkflowResult.
    Fills in safe defaults for any key that is missing.
    """
    review = result.get("review") or {}
    return {
        "ok": bool(result.get("ok", False)),
        "request_id": result.get("request_id", ""),
        "session_id": result.get("session_id", ""),
        "workflow_status": result.get("workflow_status", "FAILED"),
        "review": {
            "score": review.get("score", 0),
            "status": review.get("status", ""),
            "needs_revision": review.get("needs_revision", False),
            "feedback": review.get("feedback", []),
            "issues": review.get("issues", []),
            "dimension_scores": review.get("dimension_scores", {}),
        },
        "revision_count": result.get("revision_count", 0),
        "metadata": result.get("metadata") or {},
        "final_output": result.get("final_output") or {},
        "errors": result.get("errors") or [],
        # Workflow-specific extras (ignored by models that don't declare them)
        "email_meta": result.get("email_meta") or {},
        "seo_analysis": result.get("seo_analysis") or {},
        "social_meta": result.get("social_meta") or {},
    }
