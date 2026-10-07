"""
Content mode policy pack (Content Quality OS).

Modes:
  awareness | authority | lead_gen | seo_page

Every run resolves one mode; SEO / Review / Writer / Final QC read the pack
so rules stop fighting each other (e.g. KinvoCare in H1 vs awareness-first).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

MODES = ("awareness", "authority", "lead_gen", "seo_page")

# Default review weights (sum = 1.0). Modes may override.
_BASE_WEIGHTS: Dict[str, float] = {
    "content_quality": 0.18,
    "seo_compliance": 0.14,
    "brand_alignment": 0.18,
    "structure": 0.12,
    "factual_grounding": 0.15,
    "natural_voice": 0.18,
    "cta_effectiveness": 0.05,
}

MODE_POLICIES: Dict[str, Dict[str, Any]] = {
    "awareness": {
        "label": "Awareness",
        "goal": "Teach first; brand late",
        "h1_keyword_source": "topic_primary",
        "brand_placement": "late_third",
        "brand_in_h1": False,
        "secondary_keywords": "optional",
        "stats_quota_max": 2,
        "cta_style": "once_soft_or_brand_close",
        "hard_cta": False,
        "demonstrate_style": "family",
        "require_early_definition": False,
        "absolute_claims_strict": True,
        "review_weights": {
            **_BASE_WEIGHTS,
            "seo_compliance": 0.12,
            "natural_voice": 0.20,
        },
        "writer_notes": (
            "AWARENESS MODE: H1 and early body use topic keywords only. "
            "Introduce the brand/product name once in the final third near the CTA. "
            "Demonstrate with real family/reader scenes (hand-off, backup, sick day), "
            "not brochure definitions. Soften absolute SEO/market claims."
        ),
    },
    "authority": {
        "label": "Authority",
        "goal": "Depth + trust",
        "h1_keyword_source": "topic_primary",
        "brand_placement": "light",
        "brand_in_h1": False,
        "secondary_keywords": "optional",
        "stats_quota_max": 3,
        "cta_style": "once_soft",
        "hard_cta": False,
        "demonstrate_style": "case_research",
        "require_early_definition": True,
        "absolute_claims_strict": True,
        "review_weights": {
            **_BASE_WEIGHTS,
            "factual_grounding": 0.18,
            "seo_compliance": 0.12,
            "natural_voice": 0.16,
        },
        "writer_notes": (
            "AUTHORITY MODE: Lead with a crisp searcher definition, then topic proof "
            "and attributed research. Brand mentions stay light. Prefer case/research "
            "demonstrations. Soften unsourced absolute claims."
        ),
    },
    "lead_gen": {
        "label": "Lead gen",
        "goal": "Convert",
        "h1_keyword_source": "topic_or_brand",
        "brand_placement": "early_ok",
        "brand_in_h1": True,
        "secondary_keywords": "natural",
        "stats_quota_max": 2,
        "cta_style": "once_hard",
        "hard_cta": True,
        "demonstrate_style": "business",
        "require_early_definition": False,
        "absolute_claims_strict": False,
        "review_weights": {
            **_BASE_WEIGHTS,
            "cta_effectiveness": 0.08,
            "seo_compliance": 0.14,
            "natural_voice": 0.15,
            "content_quality": 0.17,
        },
        "writer_notes": (
            "LEAD-GEN MODE: Commercial clarity OK. One hard brand CTA at the close. "
            "Demonstrate with business outcomes (workflow, cost/time, missed deal). "
            "Sharper claims OK only when ledger-backed or clearly scoped."
        ),
    },
    "seo_page": {
        "label": "SEO page",
        "goal": "Rank for intent",
        "h1_keyword_source": "topic_primary",
        "brand_placement": "secondary",
        "brand_in_h1": False,
        "secondary_keywords": "natural",
        "stats_quota_max": 2,
        "cta_style": "intent_matched_once",
        "hard_cta": False,
        "demonstrate_style": "practical_steps",
        "require_early_definition": True,
        "absolute_claims_strict": True,
        "review_weights": {
            **_BASE_WEIGHTS,
            "seo_compliance": 0.18,
            "natural_voice": 0.14,
            "content_quality": 0.18,
        },
        "writer_notes": (
            "SEO PAGE MODE: Topic keyword in H1 when natural. Brand is secondary. "
            "Within the first 1–2 paragraphs, give a crisp searcher definition "
            "(what X is / how it differs) — not 'X is important'. Then practical steps. "
            "Soften absolute SEO/market claims unless sourced and scoped."
        ),
    },
}

def get_policy(mode: str) -> Dict[str, Any]:
    """Return a deep copy of the policy pack for a mode (fallback: seo_page)."""
    key = (mode or "").strip().lower()
    if key not in MODE_POLICIES:
        key = "seo_page"
    pack = deepcopy(MODE_POLICIES[key])
    pack["mode"] = key
    return pack


def resolve_content_mode(
    *,
    brand_context: Optional[Dict[str, Any]] = None,
    objective: str = "",
    content_mode: str = "",
    user_input: str = "",
    primary_topic: str = "",
) -> str:
    """
    Resolve content mode.

    Order:
      1. Explicit content_mode from API/UI
      2. Brand default (awareness_first → awareness)
      3. Objective (leads → lead_gen, authority → authority, seo → seo_page)
      4. Brief heuristics
      5. seo_page
    """
    explicit = (content_mode or "").strip().lower()
    if explicit in MODES:
        return explicit

    brand = brand_context if isinstance(brand_context, dict) else {}
    style = str(brand.get("content_style") or "").strip().lower()
    # Mode from brand config (content_style), never from brand name/namespace.
    if style == "awareness_first":
        # Leads objective can still force commercial mode on awareness brands
        obj = (objective or str(brand.get("objective") or "")).strip().lower()
        if obj in ("leads", "conversion", "sales"):
            return "lead_gen"
        return "awareness"

    obj = (objective or str(brand.get("objective") or "")).strip().lower()
    if obj in ("leads", "conversion", "sales"):
        return "lead_gen"
    if obj in ("authority", "thought_leadership"):
        return "authority"
    if obj in ("seo", "search"):
        return "seo_page"
    if obj in ("engagement", "awareness"):
        return "awareness"

    blob = f"{primary_topic or ''} {user_input or ''}".lower()
    if any(
        s in blob
        for s in ("book a call", "discovery call", "contact us", "get a quote", "demo")
    ):
        return "lead_gen"
    if any(s in blob for s in ("what is", "explain", "guide", "how to", "tips for")):
        return "awareness"
    if any(s in blob for s in ("report", "research", "benchmark", "industry analysis")):
        return "authority"

    return "seo_page"


def is_awareness_mode(mode_or_policy: Any) -> bool:
    if isinstance(mode_or_policy, dict):
        return str(mode_or_policy.get("mode") or "").lower() == "awareness"
    return str(mode_or_policy or "").lower() == "awareness"


def brand_allowed_in_h1(policy: Dict[str, Any]) -> bool:
    return bool(policy.get("brand_in_h1"))


def h1_uses_topic_primary(policy: Dict[str, Any]) -> bool:
    src = str(policy.get("h1_keyword_source") or "")
    return src in ("topic_primary", "topic_or_brand") and not brand_allowed_in_h1(policy)


def secondaries_enforced(policy: Dict[str, Any]) -> bool:
    return str(policy.get("secondary_keywords") or "") not in ("optional", "off", "")


def demonstrate_style(policy: Dict[str, Any]) -> str:
    return str(policy.get("demonstrate_style") or "business")


def review_weights_for(policy: Dict[str, Any]) -> Dict[str, float]:
    weights = dict(policy.get("review_weights") or _BASE_WEIGHTS)
    total = sum(weights.values()) or 1.0
    if abs(total - 1.0) > 0.01:
        weights = {k: v / total for k, v in weights.items()}
    return weights


def attach_mode_to_brand_context(
    brand_context: Dict[str, Any],
    *,
    objective: str = "",
    content_mode: str = "",
    user_input: str = "",
    primary_topic: str = "",
) -> Dict[str, Any]:
    """Non-destructive: add content_mode + mode_policy onto brand_context."""
    ctx = dict(brand_context or {})
    mode = resolve_content_mode(
        brand_context=ctx,
        objective=objective or str(ctx.get("objective") or ""),
        content_mode=content_mode or str(ctx.get("content_mode") or ""),
        user_input=user_input,
        primary_topic=primary_topic,
    )
    policy = get_policy(mode)
    ctx["content_mode"] = mode
    ctx["mode_policy"] = policy
    return ctx
