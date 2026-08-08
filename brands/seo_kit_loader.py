"""
Brand SEO kit loader
====================

Loads fixed hashtags / keywords / platform bands from brands/seo_kits.yaml.
Brand identity stays in brands/brands.yaml.
Pure config helpers — no LLM calls.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

_KIT_PATH = Path(__file__).resolve().parent / "seo_kits.yaml"

# Map our workflow (content_type, platform) → kit platform key
_PLATFORM_KEY_MAP: Dict[Tuple[str, str], str] = {
    ("linkedin", "linkedin"): "linkedin_copy",
    ("linkedin", "website"): "linkedin_copy",
    ("carousel", "linkedin"): "linkedin_copy",
    ("carousel", "carousel"): "linkedin_copy",
    ("facebook", "facebook"): "facebook_caption",
    ("instagram", "instagram"): "instagram_caption",
    ("x", "x"): "facebook_caption",
    ("twitter", "twitter"): "facebook_caption",
    ("comment", "comment"): "social_comment",
    ("reddit", "reddit"): "linkedin_copy",
    ("blog", "website"): "website_blog",
    ("blog", "blog"): "website_blog",
    ("article", "website"): "website_blog",
    ("article", "linkedin"): "linkedin_article",
    ("article", "article"): "website_blog",
    ("email", "email"): "linkedin_copy",  # length unused for email hashtags
}


@lru_cache(maxsize=1)
def _load_raw() -> Dict[str, Any]:
    if not _KIT_PATH.exists():
        logger.warning("seo_kits.yaml missing: %s", _KIT_PATH)
        return {"kits": {}, "platforms": {}}
    with open(_KIT_PATH, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    kits = data.get("kits") or {}
    # Normalise keys to lowercase for namespace lookup
    normalised = {
        str(k).lower(): v
        for k, v in kits.items()
        if isinstance(v, dict)
    }
    return {
        "kits": normalised,
        "platforms": data.get("platforms") or {},
    }


def get_kit(namespace: str) -> Dict[str, Any]:
    """Return kit dict for brand namespace (gtib, kinvo, mpm, futuristix, gcb)."""
    key = (namespace or "").strip().lower()
    kits = _load_raw()["kits"]
    return dict(kits.get(key) or {})


def resolve_platform_key(content_type: str = "", platform: str = "") -> str:
    """Map workflow content_type/platform to a kit platform band key."""
    ct = (content_type or "article").strip().lower()
    pl = (platform or "website").strip().lower()
    if pl in ("twitter",):
        pl = "x"
    # Prefer exact pair, then platform-only social shortcuts, then content type
    if (ct, pl) in _PLATFORM_KEY_MAP:
        return _PLATFORM_KEY_MAP[(ct, pl)]
    if pl in ("linkedin", "facebook", "instagram", "x", "comment", "reddit", "carousel"):
        return _PLATFORM_KEY_MAP.get((pl, pl), "website_blog")
    if ct in ("blog", "article"):
        return "website_blog"
    if ct == "comment" or pl == "comment":
        return "social_comment"
    if ct == "linkedin":
        return "linkedin_copy"
    if ct == "email":
        return "linkedin_copy"
    return "website_blog"


def get_platform_profile(
    content_type: str = "",
    platform: str = "",
) -> Dict[str, Any]:
    """Return length/density band for the resolved platform key."""
    key = resolve_platform_key(content_type, platform)
    platforms = _load_raw()["platforms"]
    profile = dict(platforms.get(key) or {})
    profile["key"] = key
    return profile


def suggested_word_count(
    content_type: str = "",
    platform: str = "",
) -> Optional[int]:
    """Midpoint of the platform band (used only when user did not set a length)."""
    profile = get_platform_profile(content_type, platform)
    lo = profile.get("min_words")
    hi = profile.get("max_words")
    try:
        if lo is not None and hi is not None:
            return int((int(lo) + int(hi)) / 2)
        if lo is not None:
            return int(lo)
        if hi is not None:
            return int(hi)
    except (TypeError, ValueError):
        return None
    return None


def _norm_tag(tag: str) -> str:
    t = str(tag or "").strip()
    if not t:
        return ""
    if not t.startswith("#"):
        t = "#" + t.lstrip("#")
    # Collapse spaces inside tags from sheet quirks
    t = re.sub(r"\s+", "", t)
    return t


def _tokens(text: str) -> set:
    return {
        t
        for t in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(t) > 2
    }


def _score_phrase(phrase: str, topic_tokens: set) -> int:
    pt = _tokens(phrase)
    if not pt or not topic_tokens:
        return 0
    return len(pt & topic_tokens)


def select_keywords_for_brief(
    kit: Dict[str, Any],
    user_input: str = "",
    primary_topic: str = "",
) -> Dict[str, List[str]]:
    """
    Pick brand/primary/secondary keywords for this brief.

    Rules (agreed):
      - Always include brand keyword when require_brand_keyword
      - Always include 1 best-matching core primary (or first primary)
      - Other primaries only if on-brief (token overlap), unless require_all_primary
      - Secondaries: all if require_all_secondary, else on-brief / top overlaps
      - Service keywords only when on-brief
    """
    if not kit:
        return {
            "brand_keywords": [],
            "primary_keywords": [],
            "secondary_keywords": [],
            "service_keywords": [],
        }

    topic = f"{primary_topic or ''} {user_input or ''}".strip()
    topic_tokens = _tokens(topic)

    brand_kws = [str(k).strip() for k in (kit.get("brand_keywords") or []) if str(k).strip()]
    primary_kws = [str(k).strip() for k in (kit.get("primary_keywords") or []) if str(k).strip()]
    secondary_kws = [str(k).strip() for k in (kit.get("secondary_keywords") or []) if str(k).strip()]
    service_kws = [str(k).strip() for k in (kit.get("service_keywords") or []) if str(k).strip()]

    selected_brand: List[str] = []
    if kit.get("require_brand_keyword", True) and brand_kws:
        selected_brand = [brand_kws[0]]

    selected_primary: List[str] = []
    if kit.get("require_all_primary"):
        selected_primary = list(primary_kws)
    elif primary_kws:
        scored = sorted(
            primary_kws,
            key=lambda p: _score_phrase(p, topic_tokens),
            reverse=True,
        )
        best = scored[0]
        selected_primary.append(best)
        for p in scored[1:]:
            if _score_phrase(p, topic_tokens) > 0:
                selected_primary.append(p)
        # Cap primaries to keep SEO natural (unless require_all)
        selected_primary = selected_primary[:4]

    selected_secondary: List[str] = []
    if kit.get("require_all_secondary"):
        selected_secondary = list(secondary_kws)
    else:
        for s in secondary_kws:
            if _score_phrase(s, topic_tokens) > 0:
                selected_secondary.append(s)
        if not selected_secondary and secondary_kws:
            # light default: top 2 by overlap, else first 2
            scored = sorted(
                secondary_kws,
                key=lambda p: _score_phrase(p, topic_tokens),
                reverse=True,
            )
            selected_secondary = scored[:2]

    selected_service: List[str] = []
    for s in service_kws:
        if _score_phrase(s, topic_tokens) > 0:
            selected_service.append(s)

    return {
        "brand_keywords": selected_brand,
        "primary_keywords": selected_primary,
        "secondary_keywords": selected_secondary,
        "service_keywords": selected_service[:3],
    }


def build_hashtags(
    kit: Dict[str, Any],
    user_input: str = "",
    platform: str = "linkedin",
    cap: int = 10,
) -> List[str]:
    """Mandatory hashtags first, then topic-matched secondary rotate tags."""
    if (platform or "").lower() in ("email", "comment"):
        return []
    if not kit:
        return []

    mandatory = [_norm_tag(t) for t in (kit.get("mandatory_hashtags") or [])]
    mandatory = [t for t in mandatory if t]
    secondary = [_norm_tag(t) for t in (kit.get("secondary_hashtags") or [])]
    secondary = [t for t in secondary if t]

    topic_tokens = _tokens(user_input)
    rotate: List[str] = []
    if secondary and topic_tokens:
        scored = sorted(
            secondary,
            key=lambda tag: _score_phrase(tag.lstrip("#"), topic_tokens),
            reverse=True,
        )
        for tag in scored:
            if _score_phrase(tag.lstrip("#"), topic_tokens) > 0:
                rotate.append(tag)
            if len(rotate) >= 3:
                break
    elif secondary:
        rotate = secondary[:2]

    # Always keep full mandatory set (marketing Excel); then fill rotate up to cap.
    out: List[str] = []
    seen = set()
    for tag in mandatory:
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(tag)
    for tag in rotate:
        if len(out) >= max(cap, len(mandatory)):
            break
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(tag)
    return out


def attach_kit_to_brand_context(
    brand_context: Dict[str, Any],
    user_input: str = "",
) -> Dict[str, Any]:
    """
    Non-destructive: add seo_kit + selected keyword lists onto brand_context.
    Safe if kit missing — returns context unchanged aside from empty kit.
    """
    ctx = dict(brand_context or {})
    namespace = str(ctx.get("namespace") or "").strip().lower()
    kit = get_kit(namespace)
    # Allow an already-attached kit (tests / overrides) to win if present
    existing = ctx.get("seo_kit")
    if isinstance(existing, dict) and existing:
        kit = existing
    ctx["seo_kit"] = kit
    if kit:
        selected = select_keywords_for_brief(
            kit,
            user_input=user_input,
            primary_topic=str(ctx.get("primary_topic") or ""),
        )
        ctx["seo_kit_selected"] = selected
        # Keep keyword_direction enriched (do not wipe existing yaml direction)
        direction = list(ctx.get("keyword_direction") or [])
        for kw in (
            selected.get("brand_keywords")
            or []
        ) + (
            selected.get("primary_keywords")
            or []
        ):
            if kw and kw not in direction:
                direction.append(kw)
        ctx["keyword_direction"] = direction
    else:
        ctx["seo_kit_selected"] = {
            "brand_keywords": [],
            "primary_keywords": [],
            "secondary_keywords": [],
            "service_keywords": [],
        }
    return ctx
