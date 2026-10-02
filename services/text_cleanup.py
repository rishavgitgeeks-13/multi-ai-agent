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
    Source / reference footers are left unchanged so titles like
    "Don't Rush, Verify First" are not mangled.
    """
    if not text:
        return text

    # Do not rewrite citation titles in the Sources footer
    split = re.split(
        r"(?=\n(?:##\s+)?Sources\b|\n(?:##\s+)?References\b)",
        text,
        maxsplit=1,
        flags=re.I,
    )
    body = split[0]
    footer = split[1] if len(split) > 1 else ""

    out = body
    for pattern, repl in _CONTRACTIONS:
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
    return out + footer


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


# Phrase → plain rewrite. Order matters (longer / more specific first).
_AI_CLICHE_REPLACEMENTS: list[tuple[str, str]] = [
    (r"(?i)\bnot only\b(.{0,100}?)\bbut also\b", r"\1, and"),
    (r"(?i)\bfuture[- ]ready\b", "prepared for what comes next"),
    (r"(?i)\bworkforce of the future\b", "teams that can adapt"),
    (r"(?i)\bdigital transformation journey\b", "shift to better systems"),
    (r"(?i)\bend[- ]to[- ]end solution\b", "complete setup"),
    (r"(?i)\bbest[- ]in[- ]class\b", "strong"),
    (r"(?i)\bstay ahead of the curve\b", "stay competitive"),
    (r"(?i)\bthe numbers are hard to ignore\b", "the numbers matter"),
    (r"(?i)\bthe payoff is clear\b", "the benefit shows up in the work"),
    (r"(?i)\bwhat matters now\b", "what to do next"),
    (r"(?i)\bchanges the game\b", "changes how the work runs"),
    (r"(?i)\bchange the game\b", "change how the work runs"),
    (r"(?i)\bunlock growth\b", "grow with less waste"),
    (r"(?i)\bdrive growth\b", "grow"),
    (r"(?i)\bscale without\b", "grow without"),
    (r"(?i)\bcompetitive advantage\b", "an edge"),
    (r"(?i)\bin today'?s market\b", "right now"),
    (r"(?i)\bholistic approach\b", "full approach"),
    (r"(?i)\bseamless integration\b", "smooth handoff"),
    (r"(?i)\bworld[- ]class\b", "high quality"),
    (r"(?i)\bcutting[- ]edge\b", "modern"),
    (r"(?i)\bgame[- ]changer\b", "real shift"),
    (r"(?i)\bgame changer\b", "real shift"),
    (r"(?i)\bleverage ai\b", "use AI"),
    (r"(?i)\bharness the power\b", "use"),
    (r"(?i)\bunlock the power\b", "get more from"),
    (r"(?i)\bMoreover,\s*", ""),
    (r"(?i)\bFurthermore,\s*", ""),
    (r"(?i)\bAdditionally,\s*", ""),
    (r"(?i)\bIn conclusion,\s*", ""),
    (r"(?i)\bIn summary,\s*", ""),
    (r"(?i)\bTo sum up,\s*", ""),
    (r"(?i)\bIt'?s worth noting that\s*", ""),
    (r"(?i)\bIt is worth noting that\s*", ""),
    (r"(?i)\bIt'?s important to note that\s*", ""),
    (r"(?i)\bIt is important to note that\s*", ""),
    (r"(?i)\bIn today'?s fast-paced world,?\s*", ""),
    (r"(?i)\bIn today'?s digital age,?\s*", ""),
    (r"(?i)\bIn the ever[- ]evolving\s+\w+,?\s*", ""),
    (r"(?i)\bIn the landscape of\s+", "In "),
    (r"(?i)\bIn the realm of\s+", "In "),
    (r"(?i)\bWhen it comes to\s+", "For "),
    (r"(?i)\bAt the end of the day,?\s*", ""),
    (r"(?i)\bNeedless to say,?\s*", ""),
    (r"(?i)\bWithout further ado,?\s*", ""),
    (r"(?i)\bThis article explores\b", "Here is a clear look at"),
    (r"(?i)\bIn this article,?\s+we will\b", "We will"),
    (r"(?i)\bLet us examine\b", "Look at"),
    (r"(?i)\bAs we delve into\b", "On"),
    (r"(?i)\bdelve into\b", "look at"),
    (r"(?i)\ba testament to\b", "proof of"),
    (r"(?i)\bplays a (?:crucial|vital|pivotal) role\b", "matters"),
    (r"(?i)\bleverage\b", "use"),
    (r"(?i)\brobust\b", "strong"),
    (r"(?i)\bseamless\b", "smooth"),
    (r"(?i)\bholistic\b", "full"),
    (r"(?i)\bnavigating the\b", "working through"),
    (r"(?i)\belevate your\b", "improve your"),
    (r"(?i)\bin essence,?\s*", ""),
    (r"(?i)\bultimately,?\s*", ""),
    (r"(?i)\brest assured,?\s*", ""),
    (r"(?i)\blook no further\.?\s*", ""),
    (r"(?i)\bthe key takeaway\b", "the point"),
    (r"(?i)\bever[- ]changing landscape\b", "shifting market"),
    (r"(?i)\bin the fast[- ]paced\b", "in the busy"),
]


def scrub_ai_cliches(text: str) -> str:
    """
    Deterministic removal/rewrite of AI / soft-B2B stock phrases.

    Shared by Writer packaging and Final Editor so flagged phrases
    do not survive into the published article.
    Leaves ## Sources / References footers untouched.
    """
    if not text:
        return text

    split = re.split(
        r"(?=\n(?:##\s+)?Sources\b|\n(?:##\s+)?References\b|\nHashtags:)",
        text,
        maxsplit=1,
        flags=re.I,
    )
    body = split[0]
    footer = split[1] if len(split) > 1 else ""

    out = body
    for pattern, repl in _AI_CLICHE_REPLACEMENTS:
        out = re.sub(pattern, repl, out)

    # Clean doubled spaces / empty sentences left by removals
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r" +([,.;:!?])", r"\1", out)
    # Re-capitalize sentence starts after phrase swaps
    out = re.sub(
        r"([.!?]\s+)([a-z])",
        lambda m: m.group(1) + m.group(2).upper(),
        out,
    )
    out = re.sub(
        r"(^|\n)([a-z])",
        lambda m: m.group(1) + m.group(2).upper(),
        out,
    )
    out = re.sub(r"\n{3,}", "\n\n", out)
    return (out.strip() + ("\n\n" + footer.lstrip("\n") if footer else "")).strip()
