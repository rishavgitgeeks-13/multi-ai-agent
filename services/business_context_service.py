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
                "facebook post",
                "thread",
            ]
        ):
            platform = "linkedin"

            if any(
                x in text
                for x in [
                    "twitter",
                    "tweet",
                    "x post",
                    "thread",
                ]
            ):
                platform = "x"

            elif "carousel" in text:
                platform = "carousel"

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

        return {
            # Flatten brand fields so Writer/Review/SEO/Research can read
            # tone, cta, display_name, namespace, etc. at the top level.
            **cfg,
            "brand_config": cfg,
            **workflow_context,
        }

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
        3. Auto-detect from user prompt (longest alias wins — avoids weak collisions).
        4. Default fallback.
        """

        # --------------------------------------------------
        # Explicit brand selection
        # --------------------------------------------------
        if brand:
            brand = brand.lower().strip()

            for cfg in self.brand_configs.values():
                aliases = [
                    alias.lower()
                    for alias in cfg.get("aliases", [])
                ]

                namespace = (
                    cfg.get("namespace", "")
                    .lower()
                )

                display_name = (
                    cfg.get("display_name", "")
                    .lower()
                )

                if (
                    brand == namespace
                    or brand == display_name
                    or brand in aliases
                    or brand in display_name
                ):
                    return self._build_context(
                        cfg,
                        user_input,
                    )

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
                aliases = [a.lower() for a in cfg.get("aliases", [])]
                namespace = str(cfg.get("namespace", "")).lower()
                display_name = str(cfg.get("display_name", "")).lower()
                if (
                    hinted == namespace
                    or hinted == display_name
                    or hinted in aliases
                    or hinted in display_name
                ):
                    return self._build_context(cfg, user_input)

        # --------------------------------------------------
        # Auto detect from prompt — longest matching alias wins
        # --------------------------------------------------
        best_cfg = None
        best_len = 0
        for cfg in self.brand_configs.values():
            for alias in cfg.get("aliases", []):
                alias_l = str(alias).lower().strip()
                if not alias_l or alias_l not in text_lower:
                    continue
                # Prefer longer, more specific aliases (e.g. "network deployment"
                # over a short token that collides across brands).
                if len(alias_l) > best_len:
                    best_cfg = cfg
                    best_len = len(alias_l)

        if best_cfg is not None:
            return self._build_context(best_cfg, user_input)

        # --------------------------------------------------
        # Fallback brand
        # --------------------------------------------------
        return self._build_context(
            self.brand_configs["futuristix"],
            user_input,
        )
