"""
Output language resolution.

Rules:
  - Manual selection (English / Hindi / Hinglish) always wins.
  - Auto-detect follows the prompt language (incl. Hinglish mix).
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
# Common Hindi/Hinglish roman tokens (lightweight heuristic)
_HINGLISH_TOKENS = {
    "hai", "hain", "ho", "kya", "kyun", "kyu", "nahi", "nahin", "mat",
    "bahut", "accha", "achha", "theek", "thik", "please", "karo", "karna",
    "chahiye", "chahie", "wala", "wali", "mein", "me", "par", "aur",
    "ka", "ki", "ke", "se", "ko", "bhi", "toh", "to", "yeh", "ye", "woh",
    "wo", "apka", "apki", "hum", "ham", "aap", "tum", "mujhe",
}
# Pure English stopwords that shouldn't alone decide English
_EN_MARKERS = {
    "the", "and", "for", "with", "this", "that", "from", "about", "into",
    "write", "create", "generate", "article", "blog", "content",
}


def detect_prompt_language(text: str) -> str:
    """
    Best-effort detect: English | Hindi | Hinglish.
    Hindi = Devanagari-heavy; Hinglish = Latin script with Hindi/roman mix.
    """
    raw = (text or "").strip()
    if not raw:
        return "English"

    dev = len(_DEVANAGARI_RE.findall(raw))
    letters = len(re.findall(r"[A-Za-z\u0900-\u097F]", raw)) or 1
    dev_ratio = dev / letters

    if dev_ratio >= 0.35:
        # Mostly Devanagari → Hindi (or Hindi with some English = still Hindi-first)
        latin = len(re.findall(r"[A-Za-z]", raw))
        if latin > 12 and latin / letters > 0.2:
            return "Hinglish"
        return "Hindi"

    tokens = {t.lower() for t in re.findall(r"[A-Za-z]+", raw)}
    hinglish_hits = sum(1 for t in tokens if t in _HINGLISH_TOKENS and len(t) > 2)
    # Strong roman-Hinglish signals
    if hinglish_hits >= 2:
        return "Hinglish"
    if hinglish_hits == 1 and len(tokens) <= 8:
        return "Hinglish"

    return "English"


def resolve_output_language(
    user_input: str,
    selected_language: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Returns (output_language, source) where source is 'manual' or 'auto'.

    Manual English/Hindi/Hinglish always overrides prompt language.
    Auto-detect / empty / 'Auto' follows the prompt.
    """
    selected = (selected_language or "").strip()
    selected_norm = selected.lower()

    if selected_norm in ("auto", "auto-detect", "detect", ""):
        detected = detect_prompt_language(user_input)
        return detected, "auto"

    # Canonicalise
    if selected_norm == "hinglish":
        return "Hinglish", "manual"
    if selected_norm == "hindi":
        return "Hindi", "manual"
    if selected_norm == "english":
        return "English", "manual"

    # Unknown label — treat as explicit manual string
    return selected.title(), "manual"


def language_writer_instruction(language: str) -> str:
    """Prompt fragment for Writer/Strategy."""
    lang = (language or "English").strip()
    if lang.lower() == "hinglish":
        return (
            "OUTPUT LANGUAGE: Hinglish (natural Hindi+English mix in Latin script "
            "and/or light Devanagari as fits). Write the full piece in Hinglish — "
            "not pure English, not pure Hindi. Keep brand CTA readable."
        )
    if lang.lower() == "hindi":
        return (
            "OUTPUT LANGUAGE: Hindi. Write the full piece in Hindi "
            "(Devanagari preferred). Keep brand names/CTAs clear."
        )
    return (
        "OUTPUT LANGUAGE: English. Write the full piece in clear professional English, "
        "even if the user prompt mixed Hindi/Hinglish."
    )
