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
from typing import Dict

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

    def resolve(self, user_input: str) -> Dict:
        """
        Identify the most appropriate business configuration
        based on the user's input.
        """

        user_input = user_input.lower()

        # Match user input with configured brand aliases.
        for brand in self.brand_configs.values():

            aliases = brand.get("aliases", [])

            if any(alias.lower() in user_input for alias in aliases):
                return brand

        # Default fallback if no brand is matched.
        raise ValueError(
            "Unable to determine the business context."
        )