"""
Shared text cleanup for generated content.

Rules:
- Remove all dash characters from article body (hyphen, en/em dash, etc.)
- Preserve URLs (hyphens inside http/https links are kept)
- Convert markdown dash bullets to asterisk bullets
- Exception: year ranges in Markdown headings only (e.g. 2020-2026)
- Expand contractions so published copy has no you're / it's / I'd style forms
"""

from __future__ import annotations

import re


_URL_RE = re.compile(r"https?://[^\s\)\]\>\"']+")
_DASH_BULLET_RE = re.compile(r"^(\s*)[-–—]\s+", re.MULTILINE)
_HR_RE = re.compile(r"^(\s*)-{3,}\s*$", re.MULTILINE)
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_HEADING_LINE_RE = re.compile(r"^(#{1,6}\s+)(.*)$", re.MULTILINE)
_YEAR_RANGE_DASH_RE = re.compile(
    r"\b((?:19|20)\d{2})\s*[-–—−]\s*((?:19|20)\d{2})\b"
)
_YEAR_RANGE_SPACE_RE = re.compile(
    r"\b((?:19|20)\d{2})\s+((?:19|20)\d{2})\b"
)

# Straight + curly apostrophe variants used in model output
_APOS = r"['’ʻʼ]"

# Longer / multi-word forms first so "wouldn't" beats "would"
_CONTRACTIONS: list[tuple[str, str]] = [
    (rf"\bwon{_APOS}t\b", "will not"),
    (rf"\bcan{_APOS}t\b", "cannot"),
    (rf"\bshan{_APOS}t\b", "shall not"),
    (rf"\bain{_APOS}t\b", "is not"),
    (rf"\bdon{_APOS}t\b", "do not"),
    (rf"\bdoesn{_APOS}t\b", "does not"),
    (rf"\bdidn{_APOS}t\b", "did not"),
    (rf"\bisn{_APOS}t\b", "is not"),
    (rf"\baren{_APOS}t\b", "are not"),
    (rf"\bwasn{_APOS}t\b", "was not"),
    (rf"\bweren{_APOS}t\b", "were not"),
    (rf"\bhasn{_APOS}t\b", "has not"),
    (rf"\bhaven{_APOS}t\b", "have not"),
    (rf"\bhadn{_APOS}t\b", "had not"),
    (rf"\bcouldn{_APOS}t\b", "could not"),
    (rf"\bwouldn{_APOS}t\b", "would not"),
    (rf"\bshouldn{_APOS}t\b", "should not"),
    (rf"\bmightn{_APOS}t\b", "might not"),
    (rf"\bmustn{_APOS}t\b", "must not"),
    (rf"\bI{_APOS}m\b", "I am"),
    (rf"\bI{_APOS}ve\b", "I have"),
    (rf"\bI{_APOS}ll\b", "I will"),
    (rf"\bI{_APOS}d\b", "I would"),
    (rf"\byou{_APOS}re\b", "you are"),
    (rf"\byou{_APOS}ve\b", "you have"),
    (rf"\byou{_APOS}ll\b", "you will"),
    (rf"\byou{_APOS}d\b", "you would"),
    (rf"\bwe{_APOS}re\b", "we are"),
    (rf"\bwe{_APOS}ve\b", "we have"),
    (rf"\bwe{_APOS}ll\b", "we will"),
    (rf"\bwe{_APOS}d\b", "we would"),
    (rf"\bthey{_APOS}re\b", "they are"),
    (rf"\bthey{_APOS}ve\b", "they have"),
    (rf"\bthey{_APOS}ll\b", "they will"),
    (rf"\bthey{_APOS}d\b", "they would"),
    (rf"\bhe{_APOS}s\b", "he is"),
    (rf"\bshe{_APOS}s\b", "she is"),
    (rf"\bit{_APOS}s\b", "it is"),
    (rf"\bthat{_APOS}s\b", "that is"),
    (rf"\bwhat{_APOS}s\b", "what is"),
    (rf"\bwho{_APOS}s\b", "who is"),
    (rf"\bwhere{_APOS}s\b", "where is"),
    (rf"\bthere{_APOS}s\b", "there is"),
    (rf"\bhere{_APOS}s\b", "here is"),
    (rf"\blet{_APOS}s\b", "let us"),
]


def expand_contractions(text: str) -> str:
    """
    Expand common English contractions to full forms.

    Brand style: published copy should read naturally without
    you're / it's / I'd apostrophe contractions.
    Possessives like "family's" are left alone.
    """
    if not text:
        return text
    out = text
    for pattern, repl in _CONTRACTIONS:
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
    return out


def strip_all_dashes(text: str) -> str:
    """
    Remove every dash-like character from generated content.

    URLs are temporarily masked so link hyphens survive.
    Year ranges inside Markdown headings keep a single ASCII hyphen
    (e.g. "# Cases 2020-2026") — body copy still has no dashes.
    """
    if not text:
        return text

    urls: list[str] = []
    year_ranges: list[str] = []

    def _mask_url(match: re.Match) -> str:
        urls.append(match.group(0))
        return f"__URL_PLACEHOLDER_{len(urls) - 1}__"

    def _mask_heading_year_range(match: re.Match) -> str:
        """Protect 2020-2026 style ranges on heading lines only."""
        prefix, rest = match.group(1), match.group(2)

        def _keep(m: re.Match) -> str:
            year_ranges.append(f"{m.group(1)}-{m.group(2)}")
            return f"__YEAR_RANGE_{len(year_ranges) - 1}__"

        rest = _YEAR_RANGE_DASH_RE.sub(_keep, rest)
        return f"{prefix}{rest}"

    masked = _URL_RE.sub(_mask_url, text)
    masked = _HEADING_LINE_RE.sub(_mask_heading_year_range, masked)
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

    for i, yr in enumerate(year_ranges):
        masked = masked.replace(f"__YEAR_RANGE_{i}__", yr)

    # If a heading already lost the dash (2020 2026), restore it there only
    def _space_years_to_dash(match: re.Match) -> str:
        prefix, rest = match.group(1), match.group(2)
        rest = _YEAR_RANGE_SPACE_RE.sub(r"\1-\2", rest)
        return f"{prefix}{rest}"

    masked = _HEADING_LINE_RE.sub(_space_years_to_dash, masked)

    for i, url in enumerate(urls):
        masked = masked.replace(f"__URL_PLACEHOLDER_{i}__", url)

    return masked.strip()
