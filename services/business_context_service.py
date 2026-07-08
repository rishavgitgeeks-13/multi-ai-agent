"""
Business Context Service
========================

Loads brand configurations from brands.yaml and identifies
which business/brand the user's request belongs to.

Responsibilities:
- Load brand configurations.
- Match user input against brand aliases.
- Return the matching brand configuration.

This service does NOT use LLMs or perform any research.
"""

from pathlib import Path
from typing import Dict, Optional

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

    def resolve(
        self,
        user_input: str = "",
        brand: Optional[str] = None,
    ) -> Dict:
        """
        Resolve the business context.

        Priority:
        1. Explicit brand selected from UI/API.
        2. Auto-detect from user prompt.
        3. Default fallback.
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
                ):
                    return cfg

        # --------------------------------------------------
        # Auto detect from prompt
        # --------------------------------------------------
        user_input = user_input.lower()

        for cfg in self.brand_configs.values():
            aliases = cfg.get("aliases", [])

            if any(
                alias.lower() in user_input
                for alias in aliases
            ):
                return cfg

        # --------------------------------------------------
        # Fallback brand
        # --------------------------------------------------
        return self.brand_configs["futuristix"]