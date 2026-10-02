"""
Business Context Service
========================

Loads brand configurations from brands.yaml and identifies
which business/brand the user's request belongs to.

Responsibilities:
- Load brand configurations.
- Match user input against brand aliases.
- Detect workflow intent (content, email, social, seo).
- Return the matching brand configuration and workflow context.

This service does NOT use LLMs or perform any research.
"""

from pathlib import Path
from typing import Dict, Optional
import re

import yaml

# Single-token aliases that are too generic alone (need a corroborating signal).
_WEAK_ALIASES = frozenset(
    {
        "property",
        "nanny",
        "telecom",
        "wireless",
        "5g",
        "mergers",
        "acquisitions",
    }
)

# Extra domain tokens that can corroborate a weak alias (namespace → tokens).
_DOMAIN_CORROBORATION = {
    "mpm": frozenset(
        {
            "nri",
            "gurgaon",
            "gurugram",
            "rera",
            "rental",
            "yield",
            "flat",
            "apartment",
            "realty",
            "landlord",
        }
    ),
    "kinvo": frozenset(
        {
            "dbs",
            "caregiver",
            "childcare",
            "babysit",
            "infant",
            "toddler",
            "japa",
            "newborn",
            "parenting",
            "nannies",
        }
    ),
    "gcb": frozenset(
        {
            "infrastructure",
            "deployment",
            "network",
            "operator",
            "tower",
            "ran",
            "fiber",
            "fibre",
        }
    ),
    "gtib": frozenset(
        {
            "founder",
            "exit",
            "sellside",
            "advisory",
            "acquisition",
            "divestiture",
        }
    ),
}


class BusinessContextService:
    """Resolves the business context for a user request."""

    def __init__(self):
        config_path = (
            Path(__file__)
            .parent.parent
            / "brands"
            / "brands.yaml"
        )

        with open(config_path, "r", encoding="utf-8") as file:
            self.brand_configs = yaml.safe_load(file)["brands"]

    @staticmethod
    def _alias_in_text(alias: str, text: str) -> bool:
        """Whole-phrase match so short tokens do not collide inside other words."""
        a = (alias or "").strip().lower()
        if not a or not text:
            return False
        pattern = r"(?<!\w)" + re.escape(a) + r"(?!\w)"
        return bool(re.search(pattern, text, flags=re.I))

    def _brand_match_score(self, cfg: Dict, text_lower: str) -> int:
        """
        Score how well a brand fits the prompt.

        Strong (multi-word / brand-named) aliases outweigh weak single tokens.
        Weak aliases alone require a second signal from the same brand.
        """
        if not text_lower:
            return 0

        strong_hits = 0
        weak_hits = 0
        best_len = 0
        weak_matched: set = set()

        for alias in cfg.get("aliases", []) or []:
            a = str(alias).lower().strip()
            if not a or not self._alias_in_text(a, text_lower):
                continue
            best_len = max(best_len, len(a))
            if a in _WEAK_ALIASES or (len(a) <= 3 and " " not in a):
                weak_hits += 1
                weak_matched.add(a)
            else:
                strong_hits += 1

        namespace = str(cfg.get("namespace") or "").lower()
        display = str(cfg.get("display_name") or "").lower()
        name_hit = False
        if namespace and self._alias_in_text(namespace, text_lower):
            name_hit = True
            strong_hits += 1
            best_len = max(best_len, len(namespace))
        # display_name may be "Kinvo Care" — match leading brand token
        display_token = display.split("/")[0].split()[0] if display else ""
        if display_token and len(display_token) >= 3 and self._alias_in_text(
            display_token, text_lower
        ):
            name_hit = True
            strong_hits += 1

        # Corroboration for weak-only matches (pain points / keyword direction).
        # Do not count the same weak alias token as its own corroboration
        # (e.g. "property tax" must not match MPM via keyword "NRI Property…").
        corroboration = 0
        if weak_hits and not strong_hits and not name_hit:
            blocked = set(weak_matched) | set(_WEAK_ALIASES)
            for phrase in list(cfg.get("keyword_direction") or []) + list(
                cfg.get("pain_points") or []
            ):
                p = str(phrase).lower().strip()
                if len(p) < 4:
                    continue
                tokens = [
                    t
                    for t in re.findall(r"[a-z0-9]{4,}", p)
                    if t not in blocked
                    and t not in {"with", "from", "that", "this", "service"}
                ]
                if sum(1 for t in tokens if t in text_lower) >= 1:
                    corroboration += 1
            ns = str(cfg.get("namespace") or "").lower()
            for tok in _DOMAIN_CORROBORATION.get(ns, ()):
                if tok not in blocked and self._alias_in_text(tok, text_lower):
                    corroboration += 1
            if corroboration == 0:
                return 0

        if strong_hits == 0 and weak_hits == 0:
            return 0

        return (strong_hits * 100) + (weak_hits * 10) + best_len + (
            corroboration * 5
        )

    def _detect_workflow(
        self,
        user_input: str,
    ) -> Dict:
        """
        Detect the workflow type and related metadata.

        Returns
        -------
        {
            "workflow": "...",
            "content_type": "...",
            "platform": "...",
            "campaign_type": "...",
            "objective": "..."
        }
        """

        text = (user_input or "").lower()

        # --------------------------------------------------
        # Social
        # --------------------------------------------------
        if any(
            x in text
            for x in [
                "linkedin",
                "twitter",
                "tweet",
                "x post",
                "social post",
                "social media",
                "carousel",
                "instagram",
                "facebook",
                "reddit",
                "comment reply",
                "social comment",
                "thread",
            ]
        ):
            platform = "linkedin"

            if any(x in text for x in ["twitter", "tweet", "x post", "thread"]):
                platform = "x"
            elif "carousel" in text:
                platform = "carousel"
            elif "instagram" in text:
                platform = "instagram"
            elif "facebook" in text:
                platform = "facebook"
            elif "reddit" in text:
                platform = "reddit"
            elif any(
                x in text
                for x in [
                    "comment reply",
                    "social comment",
                    "reply comment",
                    "comment on this",
                    "write a comment",
                    "post a comment",
                ]
            ):
                platform = "comment"

            return {
                "workflow": "social",
                "content_type": None,
                "platform": platform,
                "campaign_type": None,
                "objective": "engagement",
            }

        # --------------------------------------------------
        # Email
        # --------------------------------------------------
        if any(
            x in text
            for x in [
                "email",
                "newsletter",
                "cold email",
                "mail",
                "email campaign",
                "drip campaign",
                "promotional email",
            ]
        ):
            campaign_type = "promotional"

            if "newsletter" in text:
                campaign_type = "newsletter"

            elif "transactional" in text:
                campaign_type = "transactional"

            elif "nurture" in text:
                campaign_type = "nurture"

            return {
                "workflow": "email",
                "content_type": None,
                "platform": None,
                "campaign_type": campaign_type,
                "objective": "leads",
            }

        # --------------------------------------------------
        # SEO
        # --------------------------------------------------
        if any(
            x in text
            for x in [
                "seo analysis",
                "keyword research",
                "search intent",
                "meta description",
                "seo strategy",
                "seo audit",
                "keywords for",
                "ranking keywords",
            ]
        ):
            return {
                "workflow": "seo",
                "content_type": "article",
                "platform": None,
                "campaign_type": None,
                "objective": "seo",
            }

        # --------------------------------------------------
        # Default: Content
        # --------------------------------------------------
        content_type = "article"

        if "blog" in text:
            content_type = "blog"

        return {
            "workflow": "content",
            "content_type": content_type,
            "platform": None,
            "campaign_type": None,
            "objective": "seo",
        }

    def _build_context(
        self,
        cfg: Dict,
        user_input: str,
    ) -> Dict:
        """
        Build final context payload.
        """

        workflow_context = self._detect_workflow(
            user_input
        )

        context = {
            # Flatten brand fields so Writer/Review/SEO/Research can read
            # tone, cta, display_name, namespace, etc. at the top level.
            **cfg,
            "brand_config": cfg,
            **workflow_context,
        }
        # Attach fixed hashtag/keyword kit from brands/seo_kits.yaml.
        # Soft-fail: missing kit must not break brand resolution.
        try:
            from brands.seo_kit_loader import attach_kit_to_brand_context

            context = attach_kit_to_brand_context(context, user_input=user_input)
        except Exception:
            context.setdefault("seo_kit", {})
            context.setdefault("seo_kit_selected", {})
        return context

    def _cfg_matches_brand_label(self, cfg: Dict, brand: str) -> bool:
        brand = (brand or "").lower().strip()
        if not brand:
            return False
        aliases = [alias.lower() for alias in cfg.get("aliases", [])]
        namespace = str(cfg.get("namespace", "")).lower()
        display_name = str(cfg.get("display_name", "")).lower()
        return (
            brand == namespace
            or brand == display_name
            or brand in aliases
            or brand in display_name
        )

    def resolve(
        self,
        user_input: str = "",
        brand: Optional[str] = None,
    ) -> Dict:
        """
        Resolve the business context.

        Priority:
        1. Explicit brand selected from UI/API.
        2. `[Brand: …]` hint embedded in the user prompt.
        3. Auto-detect from user prompt (strongest scored match).
        4. Default fallback.
        """

        # --------------------------------------------------
        # Explicit brand selection
        # --------------------------------------------------
        if brand:
            brand_l = brand.lower().strip()
            for cfg in self.brand_configs.values():
                if self._cfg_matches_brand_label(cfg, brand_l):
                    return self._build_context(cfg, user_input)

        text = (user_input or "").strip()
        text_lower = text.lower()

        # --------------------------------------------------
        # Embedded [Brand: …] hint (workflows often prefix this)
        # --------------------------------------------------
        brand_hint = re.search(
            r"\[\s*brand\s*:\s*([^\]]+)\]",
            text,
            flags=re.I,
        )
        if brand_hint:
            hinted = brand_hint.group(1).strip().lower()
            for cfg in self.brand_configs.values():
                if self._cfg_matches_brand_label(cfg, hinted):
                    return self._build_context(cfg, user_input)

        # --------------------------------------------------
        # Auto detect from prompt — highest match score wins
        # --------------------------------------------------
        best_cfg = None
        best_score = 0
        for cfg in self.brand_configs.values():
            score = self._brand_match_score(cfg, text_lower)
            if score > best_score:
                best_cfg = cfg
                best_score = score

        if best_cfg is not None and best_score > 0:
            return self._build_context(best_cfg, user_input)

        # --------------------------------------------------
        # Fallback brand
        # --------------------------------------------------
        return self._build_context(
            self.brand_configs["futuristix"],
            user_input,
        )
