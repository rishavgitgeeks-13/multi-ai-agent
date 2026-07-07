"""
Brand Loader
============

Loads all brand configurations from brands.yaml.

Responsibilities:
- Load brands.yaml
- Return all brand configurations
- Return a specific brand configuration

This module only reads configuration.
It does not perform any business logic.
"""

from pathlib import Path
from typing import Dict, Optional

import yaml


class BrandLoader:
    """Loads brand configurations from YAML."""

    def __init__(self, config_path: Optional[str] = None):
        if config_path:
            self.config_path = Path(config_path)
        else:
            self.config_path = (
                Path(__file__).resolve().parent.parent
                / "brands"
                / "brands.yaml"
            )

        self._brands = self._load()

    def _load(self) -> Dict:
        """Load all brand configurations."""

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Brand configuration not found: {self.config_path}"
            )

        with open(self.config_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        return data.get("brands", {})

    def get_all_brands(self) -> Dict:
        """Return all configured brands."""
        return self._brands

    def get_brand(self, brand_name: str) -> Optional[Dict]:
        """Return a specific brand configuration."""
        return self._brands.get(brand_name)

    def brand_exists(self, brand_name: str) -> bool:
        """Check whether a brand exists."""
        return brand_name in self._brands