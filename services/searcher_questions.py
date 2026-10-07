"""
Searcher Question Pack (information gain)
=========================================

Builds 5–8 PAA-style questions from the brief + research so Strategy/Writer
outline around what the searcher actually asks — not only the keyword.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_STOP = frozenset(
    {
        "a", "an", "the", "and", "or", "for", "to", "of", "in", "on", "with",
        "how", "what", "which", "when", "where", "why", "who", "write", "article",
        "blog", "guide", "about", "from", "into", "your", "our", "this", "that",
        "using", "please", "need", "make", "create",
    }
)


def _topic_phrase(user_input: str, primary_topic: str = "") -> str:
    raw = (primary_topic or user_input or "").strip()
    raw = re.sub(
        r"(?i)^(write|create|draft|generate)\s+(an?\s+)?(article|blog|guide|post)\s+(on|about|for)?\s*",
        "",
        raw,
    ).strip()
    raw = re.sub(r"\s+", " ", raw)
    return raw[:120] if raw else (user_input or "this topic")[:80]


def _tokens(text: str) -> List[str]:
    return [
        t
        for t in re.findall(r"[a-z0-9]{3,}", (text or "").lower())
        if t not in _STOP
    ][:12]


def build_searcher_question_pack(
    *,
    user_input: str,
    primary_topic: str = "",
    research_data: Optional[Dict[str, Any]] = None,
    max_questions: int = 8,
) -> List[str]:
    """
    Return 5–8 searcher questions the article should answer.

    Heuristic (no LLM): brief shape + vs/cost/how-to + research cues.
    """
    topic = _topic_phrase(user_input, primary_topic)
    ui = (user_input or "").strip()
    ui_l = ui.lower()
    toks = _tokens(f"{topic} {ui}")
    core = " ".join(toks[:5]) or topic

    questions: List[str] = []
    seen: set[str] = set()

    def _add(q: str) -> None:
        q = re.sub(r"\s+", " ", (q or "").strip())
        if not q or not q.endswith("?"):
            if q and not q.endswith("?"):
                q = q.rstrip(".") + "?"
        key = q.lower()
        if len(q) < 12 or key in seen:
            return
        seen.add(key)
        questions.append(q)

    # If the brief is already a question, lead with it
    if "?" in ui:
        first_q = ui.split("?")[0].strip() + "?"
        _add(first_q)

    # Definition / distinction
    _add(f"What is {topic}?")
    if re.search(r"(?i)\bvs\.?\b|\bversus\b|\bor\b", ui_l):
        _add(f"How do you choose between options for {core}?")
    else:
        _add(f"How is {topic} different from common alternatives?")

    # Decision / how-to
    _add(f"How do you choose the right approach for {core}?")
    _add(f"What should you check before deciding on {core}?")

    # Cost / risk / failure modes when relevant
    if re.search(r"(?i)\b(cost|price|salary|fee|budget|roi)\b", ui_l):
        _add(f"What does {core} typically cost, and what drives the price?")
    if re.search(r"(?i)\b(risk|safe|scam|fraud|abuse|vetting|trust)\b", ui_l):
        _add(f"What are the main risks with {core}, and how do you reduce them?")
    else:
        _add(f"What mistakes do people make with {core}?")

    # Audience / place cues from brief
    if re.search(r"(?i)\bnri\b|non[-\s]?resident", ui_l):
        _add(f"What changes for NRI families when evaluating {core}?")
    if re.search(r"(?i)\b(gurgaon|gurugram|noida|delhi|mumbai|london|uk|india)\b", ui_l):
        m = re.search(
            r"(?i)\b(gurgaon|gurugram|noida|delhi|mumbai|london|uk|india)\b", ui_l
        )
        place = m.group(1) if m else "this market"
        _add(f"What is different about {core} in {place}?")

    # Research-driven extras (titles that look like Qs)
    research = research_data or {}
    for doc in (research.get("documents") or [])[:8]:
        if isinstance(doc, dict):
            title = str(doc.get("title") or "")
        else:
            title = str(getattr(doc, "title", "") or "")
        if "?" in title:
            _add(title.strip()[:140])
        elif re.search(r"(?i)^(how|what|why|when|which)\b", title.strip()):
            _add(title.strip().rstrip(".") + "?")

    # SERP differentiation prompt-as-question
    _add(f"What do most articles miss about {core}?")

    out = questions[: max(5, min(max_questions, 8))]
    # Ensure at least 5
    fillers = [
        f"Who is {topic} for?",
        f"When does {topic} make sense?",
        f"What does a good process for {core} look like?",
    ]
    for f in fillers:
        if len(out) >= 5:
            break
        _add(f)
        out = questions[: max(5, min(max_questions, 8))]

    logger.info("Searcher question pack | n=%d | topic=%s", len(out), topic[:60])
    return out


def format_questions_for_outline(questions: List[str]) -> str:
    if not questions:
        return ""
    lines = [
        "SEARCHER QUESTION PACK (information gain — outline must answer these):",
        "Map major H2 sections to these questions. Prefer decision criteria, "
        "checklists, or comparison tables over repeated thesis paragraphs.",
        "SERP DIFFERENTIATION: include at least one section that adds what most "
        "ranking articles miss (local rule, failure mode, decision table, or checklist).",
    ]
    for i, q in enumerate(questions, 1):
        lines.append(f"  Q{i}. {q}")
    return "\n".join(lines)


def question_coverage_issues(
    draft: str,
    questions: Optional[List[str]] = None,
    min_covered: int = 3,
) -> List[str]:
    """Soft review: flag when too few searcher questions appear answered in headings/body."""
    qs = [q for q in (questions or []) if q]
    if len(qs) < 3:
        return []
    body = (draft or "").lower()
    if len(body) < 400:
        return []

    covered = 0
    for q in qs:
        toks = _tokens(q)
        if len(toks) < 2:
            continue
        # Count as covered if ≥2 distinctive tokens appear
        hits = sum(1 for t in toks[:6] if t in body)
        if hits >= 2:
            covered += 1

    if covered < min_covered:
        return [
            f"INFO_GAIN: Only ~{covered}/{len(qs)} searcher questions look answered. "
            "Expand sections so the outline answers the Searcher Question Pack "
            "(definition, choice criteria, risks/mistakes, and what most articles miss)."
        ]
    return []
