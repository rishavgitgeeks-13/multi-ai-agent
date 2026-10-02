"""
Final QC detectors (editorial mechanics).

Catches the recurring ~8.2 defects that Review scorecards miss:
  - SEO keyword over-injection
  - Thesis repetition across sections
  - Soft AI/B2B template phrases
  - Ornamental / authority-theater stats
  - Duplicate CTAs
  - Formulaic section openings

These flags feed the Final Editor surgical pass.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence


SOFT_B2B_PHRASES: List[str] = [
    "future-ready",
    "future ready",
    "workforce of the future",
    "changes the game",
    "change the game",
    "the payoff is clear",
    "the numbers are hard to ignore",
    "what matters now",
    "unlock growth",
    "drive growth",
    "scale without",
    "competitive advantage",
    "stay ahead of the curve",
    "in today's market",
    "digital transformation journey",
    "holistic approach",
    "seamless integration",
    "end-to-end solution",
    "best-in-class",
    "best in class",
    "world-class",
    "cutting-edge",
    "game-changer",
    "game changer",
    "leverage ai",
    "harness the power",
    "unlock the power",
]


def _body_only(draft: str) -> str:
    text = draft or ""
    return re.split(
        r"(?=\n(?:##\s+)?Sources\b|\n(?:##\s+)?References\b|\nHashtags:)",
        text,
        maxsplit=1,
        flags=re.I,
    )[0]


def _split_h2_sections(body: str) -> List[tuple[str, str]]:
    """Return list of (heading, section_body) for ## sections."""
    parts = re.split(r"(?im)^##\s+", body or "")
    if len(parts) <= 1:
        return []
    out: List[tuple[str, str]] = []
    for chunk in parts[1:]:
        lines = chunk.split("\n", 1)
        heading = (lines[0] or "").strip()
        sec = lines[1] if len(lines) > 1 else ""
        out.append((heading, sec))
    return out


def keyword_density_issues(
    draft: str,
    primary: Optional[Sequence[str]] = None,
    secondary: Optional[Sequence[str]] = None,
) -> List[str]:
    """Flag over-injected primary/secondary keywords in long-form copy."""
    issues: List[str] = []
    body = _body_only(draft)
    words = body.split()
    n = max(len(words), 1)
    lower = body.lower()

    for kw in list(primary or [])[:4]:
        term = (kw or "").strip().lower()
        if len(term) < 4:
            continue
        count = lower.count(term)
        # More than ~1.8% density or >8 raw hits on a mid article → stuffing
        density = (count * max(len(term.split()), 1)) / n
        if count >= 8 or (n >= 400 and density > 0.018 and count >= 5):
            issues.append(
                f"QC_SEO_STUFF: Primary term \"{kw}\" appears {count} times "
                f"(~{density * 100:.1f}% density). Thin repeated uses; keep 2–4 natural mentions."
            )

    for kw in list(secondary or [])[:8]:
        term = (kw or "").strip().lower()
        if len(term) < 5:
            continue
        count = lower.count(term)
        if count >= 5:
            issues.append(
                f"QC_SEO_STUFF: Secondary term \"{kw}\" appears {count} times. "
                "Remove forced injections; keep at most 1–2 natural uses."
            )
    return issues


def thesis_repetition_issues(draft: str) -> List[str]:
    """
    Detect near-repeated thesis sentences across H2 sections.

    Uses normalized 8–14 word windows; repeated windows across different
    sections indicate the Writer restating the same core claim.
    """
    issues: List[str] = []
    sections = _split_h2_sections(_body_only(draft))
    if len(sections) < 3:
        return issues

    def _norm_sentences(text: str) -> List[str]:
        raw = re.split(r"(?<=[.!?])\s+", text or "")
        out = []
        for s in raw:
            s = re.sub(r"[^a-z0-9\s]", " ", s.lower())
            s = re.sub(r"\s+", " ", s).strip()
            toks = s.split()
            if 8 <= len(toks) <= 28:
                out.append(" ".join(toks))
        return out

    # Collect 6-grams per section
    gram_owner: Dict[str, str] = {}
    repeats: List[str] = []
    for heading, sec in sections:
        h_l = heading.lower()
        if h_l.startswith(("conclusion", "next step", "sources", "faq")):
            continue
        for sent in _norm_sentences(sec):
            toks = sent.split()
            for i in range(0, max(0, len(toks) - 5)):
                gram = " ".join(toks[i : i + 6])
                if gram in gram_owner and gram_owner[gram] != heading:
                    if gram not in repeats:
                        repeats.append(gram)
                else:
                    gram_owner[gram] = heading

    if len(repeats) >= 3:
        sample = "; ".join(f'"{r}"' for r in repeats[:3])
        issues.append(
            "QC_THESIS_REPEAT: Core claims are restated across sections "
            f"({len(repeats)} overlapping phrases). Keep the thesis once; "
            f"later sections must advance a new angle. Samples: {sample}."
        )
    return issues


def soft_b2b_phrase_issues(draft: str) -> List[str]:
    """Flag soft B2B / template phrases that classic AI-tell lists miss."""
    lower = _body_only(draft).lower()
    hits = [p for p in SOFT_B2B_PHRASES if p in lower]
    if len(hits) >= 3:
        return [
            "QC_B2B_CLICHE: Soft template phrases still present ("
            + ", ".join(f'"{h}"' for h in hits[:6])
            + "). Rewrite in plain operator language."
        ]
    if hits:
        return [
            "QC_B2B_CLICHE: Remove soft B2B template phrase "
            f'"{hits[0]}" and rewrite specifically.'
        ]
    return []


def ornamental_stat_issues(draft: str, primary_topic: str = "") -> List[str]:
    """
    Soft heuristic: many attributed stats that don't share tokens with the brief
    look like authority padding.
    """
    issues: List[str] = []
    body = _body_only(draft)
    brief_tokens = {
        t
        for t in re.findall(r"[a-z0-9]{4,}", (primary_topic or "").lower())
        if t
        not in {
            "with",
            "from",
            "that",
            "this",
            "about",
            "write",
            "article",
            "blog",
            "using",
            "into",
            "your",
            "have",
            "will",
        }
    }
    # "According to X: 78%" style
    attributions = re.findall(
        r"(?i)(?:according to|as per|reports? that)\s+([^.\n]{8,120})",
        body,
    )
    if len(attributions) < 3 or not brief_tokens:
        return issues

    off = 0
    for attr in attributions:
        a_l = attr.lower()
        if not any(tok in a_l for tok in brief_tokens):
            # Also check following 120 chars for brief tokens
            idx = body.lower().find(attr.lower())
            window = body.lower()[idx : idx + 160] if idx >= 0 else a_l
            if not any(tok in window for tok in brief_tokens):
                off += 1
    if off >= 2:
        issues.append(
            f"QC_ORNAMENTAL_STAT: {off} attributed statistics look off-angle vs the brief. "
            "Keep only section-relevant proof; drop ornamental authority numbers."
        )
    return issues


def duplicate_cta_issues(draft: str, cta: str = "") -> List[str]:
    """Flag more than one hard CTA / discovery-call style close."""
    issues: List[str] = []
    body = _body_only(draft)
    lower = body.lower()
    cta_l = (cta or "").strip().lower()

    count = 0
    if cta_l and len(cta_l) >= 8:
        count = lower.count(cta_l)
    else:
        # Generic commercial CTA patterns
        patterns = [
            r"book an? ai discovery call",
            r"book a(?:n)? (?:free )?discovery call",
            r"schedule a(?:n)? (?:demo|call|consultation)",
            r"contact (?:us|our team)\b",
        ]
        for p in patterns:
            count += len(re.findall(p, lower))

    if count >= 2:
        issues.append(
            f"QC_CTA_DUP: CTA appears {count} times. Keep a single closing CTA "
            "(prefer Conclusion); remove duplicates from Next Steps / mid-article."
        )
    return issues


def formulaic_structure_issues(draft: str) -> List[str]:
    """Detect cloned section openings (same scaffold across 3+ H2s)."""
    issues: List[str] = []
    sections = _split_h2_sections(_body_only(draft))
    if len(sections) < 4:
        return issues

    lead_patterns = []
    for heading, sec in sections:
        h_l = heading.lower()
        if h_l.startswith(("conclusion", "next step", "sources")):
            continue
        # First non-empty paragraph
        paras = [p.strip() for p in re.split(r"\n\s*\n", sec) if p.strip()]
        if not paras:
            continue
        lead = re.sub(r"\s+", " ", paras[0])[:90].lower()
        lead = re.sub(r"[^a-z0-9\s]", "", lead)
        # Normalize to first 6 tokens as shape fingerprint
        toks = lead.split()[:6]
        if len(toks) >= 4:
            lead_patterns.append(" ".join(toks))

    if not lead_patterns:
        return issues
    counts = Counter(lead_patterns)
    # Also catch shared scaffolding starts
    scaffold_hits = 0
    for lead in lead_patterns:
        if re.match(
            r"^(the|this|for|in|when|outsourcing|ai|businesses|teams|companies)\b",
            lead,
        ):
            # Weak signal alone — need identical fingerprints
            pass
    dupes = [p for p, c in counts.items() if c >= 2]
    if len(dupes) >= 2 or any(c >= 3 for c in counts.values()):
        issues.append(
            "QC_FORMULA: Multiple sections open with the same template rhythm. "
            "Vary openings (scene, number, objection, question) — avoid identical "
            "claim→example→transition blocks."
        )
    return issues


def mode_policy_issues(
    draft: str,
    mode_policy: Optional[Dict[str, Any]] = None,
    brand_keywords: Optional[Sequence[str]] = None,
) -> List[str]:
    """Flag mode violations (e.g. brand in H1 on awareness)."""
    issues: List[str] = []
    policy = mode_policy if isinstance(mode_policy, dict) else {}
    if not policy:
        return issues
    if policy.get("brand_in_h1"):
        return issues
    h1 = re.search(r"^#\s+(.+)$", draft or "", re.MULTILINE)
    if not h1:
        return issues
    h1_l = h1.group(1).lower()
    for bk in list(brand_keywords or [])[:3]:
        term = (bk or "").strip().lower()
        if len(term) < 3:
            continue
        # Fair match: spaced or compound
        compact = term.replace(" ", "")
        if term in h1_l or compact in h1_l.replace(" ", ""):
            issues.append(
                f"QC_MODE_H1: Brand term \"{bk}\" appears in H1 but mode "
                f"{policy.get('mode') or ''} forbids brand_in_h1. Use a topic H1."
            )
            break
    return issues


def run_final_qc(
    draft: str,
    *,
    primary_keywords: Optional[Sequence[str]] = None,
    secondary_keywords: Optional[Sequence[str]] = None,
    cta: str = "",
    primary_topic: str = "",
    content_type: str = "article",
    mode_policy: Optional[Dict[str, Any]] = None,
    brand_keywords: Optional[Sequence[str]] = None,
) -> List[str]:
    """Run all Final QC detectors for long-form content."""
    ct = (content_type or "").lower()
    if ct not in ("blog", "article"):
        # Still check CTA + soft B2B on medium-form
        flags: List[str] = []
        flags.extend(soft_b2b_phrase_issues(draft))
        flags.extend(duplicate_cta_issues(draft, cta=cta))
        flags.extend(
            mode_policy_issues(draft, mode_policy=mode_policy, brand_keywords=brand_keywords)
        )
        return flags

    flags = []
    flags.extend(keyword_density_issues(draft, primary_keywords, secondary_keywords))
    flags.extend(thesis_repetition_issues(draft))
    flags.extend(soft_b2b_phrase_issues(draft))
    flags.extend(ornamental_stat_issues(draft, primary_topic=primary_topic))
    flags.extend(duplicate_cta_issues(draft, cta=cta))
    flags.extend(formulaic_structure_issues(draft))
    flags.extend(
        mode_policy_issues(draft, mode_policy=mode_policy, brand_keywords=brand_keywords)
    )
    try:
        from services.fidelity_gate import humanize_review_issues

        flags.extend(humanize_review_issues(draft))
    except Exception:
        pass
    try:
        from services.junk_gate import junk_issues

        flags.extend(junk_issues(draft))
    except Exception:
        pass
    return flags


def qc_summary(flags: List[str]) -> Dict[str, Any]:
    return {
        "flag_count": len(flags or []),
        "flags": list(flags or []),
        "needs_edit": bool(flags),
    }
