"""
Shared CTA policy for Writer + Review.

Keeps commercial pieces on a hard brand CTA, while awareness-first /
educational briefs stay soft so Review does not force a sales close
that fights the Writer.

Default behaviour for GTIB / MPM / Futuristix / GCB commercial articles
is unchanged: hard CTA still required.
"""

from __future__ import annotations

from typing import Any, Dict


def _mode_policy(brand_context: Dict[str, Any] | None) -> Dict[str, Any]:
    brand = brand_context if isinstance(brand_context, dict) else {}
    policy = brand.get("mode_policy")
    if isinstance(policy, dict) and policy:
        return policy
    mode = str(brand.get("content_mode") or "").strip().lower()
    if mode:
        try:
            from config.mode_policies import get_policy

            return get_policy(mode)
        except Exception:
            return {}
    return {}


def is_awareness_first(brand_context: Dict[str, Any] | None) -> bool:
    """True for awareness mode / brands with content_style=awareness_first."""
    brand = brand_context if isinstance(brand_context, dict) else {}
    policy = _mode_policy(brand)
    mode = str(policy.get("mode") or brand.get("content_mode") or "").lower()
    if mode == "awareness":
        return True
    if mode in ("lead_gen", "seo_page", "authority"):
        return False
    style = str(brand.get("content_style") or "").strip().lower()
    return style == "awareness_first"


def cta_is_hard_required(
    content_type: str = "",
    primary_topic: str = "",
    awareness_first: bool = False,
    objective: str = "",
    brand_context: Dict[str, Any] | None = None,
) -> bool:
    """
    Hard verbatim CTA only for commercial long-form / lead-gen intents.

    Soft (False) for comments, carousels, explain/how-to briefs, and
    awareness-first brands on non-sales objectives.
    """
    policy = _mode_policy(brand_context)
    if policy:
        # Mode pack is authoritative when present
        if "hard_cta" in policy:
            ct = (content_type or "").lower()
            if ct in ("comment", "carousel"):
                return False
            return bool(policy.get("hard_cta"))

    if brand_context and not awareness_first:
        awareness_first = is_awareness_first(brand_context)

    ct = (content_type or "").lower()
    if ct in ("comment", "carousel"):
        return False

    obj = (objective or "").lower()
    if obj in ("leads", "conversion", "sales"):
        return True

    topic = (primary_topic or "").lower()
    soft_signals = (
        "explain",
        "what is",
        "what are",
        "guide",
        "how to",
        "tips",
        "thank",
        "reply",
        "feedback",
        "comment",
        "overview",
        "meaning of",
    )
    if any(s in topic for s in soft_signals):
        return False

    if awareness_first and obj in (
        "",
        "seo",
        "authority",
        "engagement",
        "awareness",
    ):
        return False

    return ct in ("blog", "article", "email", "linkedin")


def review_cta_instruction(
    cta: str,
    hard_required: bool,
) -> str:
    """Prompt snippet for Review so soft CTAs are not scored as failures."""
    cta_line = (cta or "").strip()
    if not cta_line:
        return "REQUIRED CTA: none"
    if hard_required:
        return (
            f"REQUIRED CTA (hard — must appear near the close, verbatim or "
            f"clear brand-named close): {cta_line}"
        )
    return (
        f"CTA POLICY (soft — do NOT fail solely for missing verbatim CTA): "
        f"prefer a natural close; if a CTA fits, use \"{cta_line}\" late in "
        f"the piece. Educational / awareness drafts may omit a hard sell."
    )
