"""
Shared text cleanup for generated content.

Rules:
- Remove all dash characters from article body (hyphen, en/em dash, etc.)
- Preserve URLs (hyphens inside http/https links are kept)
- Convert markdown dash bullets to asterisk bullets
"""

from __future__ import annotations

import re


_URL_RE = re.compile(r"https?://[^\s\)\]\>\"']+")
_DASH_BULLET_RE = re.compile(r"^(\s*)[-–—]\s+", re.MULTILINE)
_HR_RE = re.compile(r"^(\s*)-{3,}\s*$", re.MULTILINE)
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")


def strip_all_dashes(text: str) -> str:
    """
    Remove every dash-like character from generated content.

    URLs are temporarily masked so link hyphens survive.
    """
    if not text:
        return text

    urls: list[str] = []

    def _mask(match: re.Match) -> str:
        urls.append(match.group(0))
        return f"__URL_PLACEHOLDER_{len(urls) - 1}__"

    masked = _URL_RE.sub(_mask, text)
    masked = _DASH_BULLET_RE.sub(r"\1* ", masked)
    masked = _HR_RE.sub("", masked)

    # Unicode dashes / minus signs → space or connector words
    for ch in (
        "\u2014",  # em dash —
        "\u2013",  # en dash –
        "\u2212",  # minus −
        "\u2012",  # figure dash
        "\u2010",  # hyphen
        "\u2011",  # non-breaking hyphen
        "\ufe58",  # small em dash
        "\ufe63",  # small hyphen-minus
        "\uff0d",  # fullwidth hyphen-minus
    ):
        masked = masked.replace(ch, " ")

    # ASCII hyphen-minus
    masked = masked.replace("-", " ")
    masked = _MULTI_SPACE_RE.sub(" ", masked)
    # Clean spaces before punctuation introduced by replacements
    masked = re.sub(r" +([,.;:!?])", r"\1", masked)
    masked = re.sub(r"\n{3,}", "\n\n", masked)

    for i, url in enumerate(urls):
        masked = masked.replace(f"__URL_PLACEHOLDER_{i}__", url)

    return masked.strip()
