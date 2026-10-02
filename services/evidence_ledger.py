"""
Evidence ledger (generic — all brands)
======================================

Research → structured claim pack → Writer may only use these facts.
Sources footer prefers ledger URLs. Review fails unsupported numbers.

Each entry:
  id | kind | statement | figure | year | geography | source_title | url | confidence
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Set

logger = logging.getLogger(__name__)

_FIGURE_RE = re.compile(
    r"("
    r"\d+(?:\.\d+)?\s*%|"
    r"(?:INR|Rs\.?|₹|USD|GBP|EUR)\s*\d[\d,]*(?:\.\d+)?(?:\s*(?:crore|lakh|million|billion))?|"
    r"\$\d[\d,]*(?:\.\d+)?|"
    r"\d[\d,]*(?:\.\d+)?\s*(?:million|billion|thousand|crore|lakh|cases?|complaints?|"
    r"incidents?|arrests?|nannies|caregivers?|families|parents|children)|"
    r"\b\d{1,3}(?:,\d{3})+\b"
    r")",
    re.I,
)
_YEAR_RE = re.compile(r"\b(20[12]\d)\b")
_TOOL_LABEL = re.compile(
    r"\b(tavily|duckduckgo|newsapi|research summary|web search)\b",
    re.I,
)


def _norm_figure(raw: str) -> str:
    return re.sub(r"\s+", " ", (raw or "").strip().lower())


def _figures_in(text: str) -> Set[str]:
    return {_norm_figure(m.group(0)) for m in _FIGURE_RE.finditer(text or "")}


def _years_in(text: str) -> List[str]:
    return list(dict.fromkeys(_YEAR_RE.findall(text or "")))


def _guess_geo(text: str) -> str:
    t = (text or "").lower()
    checks = [
        ("delhi", "Delhi NCR"),
        ("gurgaon", "Gurgaon"),
        ("gurugram", "Gurgaon"),
        ("bengaluru", "Bengaluru"),
        ("bangalore", "Bengaluru"),
        ("mumbai", "Mumbai"),
        ("london", "UK"),
        ("united kingdom", "UK"),
        ("britain", "UK"),
        ("pakistan", "Pakistan"),
        ("california", "US"),
        ("united states", "US"),
        ("india", "India"),
        ("uk", "UK"),
        ("usa", "US"),
    ]
    for needle, label in checks:
        if re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", t):
            return label
    return ""


def build_evidence_ledger(
    *,
    documents: Optional[List[Any]] = None,
    sources: Optional[List[Any]] = None,
    statistics: Optional[List[str]] = None,
    incidents: Optional[List[str]] = None,
    brief_lock: Optional[Dict[str, Any]] = None,
    max_entries: int = 14,
) -> List[Dict[str, Any]]:
    """
    Build a claim pack from research artifacts.

    Soft-filters with BriefLock when present (on-brief / on-market).
    """
    lock = None
    if brief_lock:
        try:
            from services.fidelity_gate import BriefLock, evidence_is_on_brief

            lock = BriefLock.from_dict(brief_lock)
        except Exception:
            lock = None
            evidence_is_on_brief = None  # type: ignore
    else:
        evidence_is_on_brief = None  # type: ignore

    # Index sources by normalised title for URL attachment
    url_by_title: Dict[str, str] = {}
    source_rows: List[Dict[str, str]] = []
    for src in sources or []:
        if isinstance(src, dict):
            title = str(src.get("title") or src.get("name") or "").strip()
            url = str(src.get("url") or "").strip()
        else:
            title = str(getattr(src, "title", "") or "").strip()
            url = str(getattr(src, "url", "") or "").strip()
        if not title and not url:
            continue
        if _TOOL_LABEL.search(title):
            continue
        source_rows.append({"title": title, "url": url})
        if title:
            url_by_title[re.sub(r"\s+", " ", title.lower())[:80]] = url

    # Also index documents
    for doc in documents or []:
        if isinstance(doc, dict):
            title = str(doc.get("title") or "").strip()
            url = str(doc.get("url") or "").strip()
        else:
            title = str(getattr(doc, "title", "") or "").strip()
            url = str(getattr(doc, "url", "") or "").strip()
        if title and url:
            url_by_title.setdefault(re.sub(r"\s+", " ", title.lower())[:80], url)

    entries: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    def _add(
        kind: str,
        statement: str,
        source_title: str = "",
        url: str = "",
        confidence: float = 0.6,
    ) -> None:
        stmt = re.sub(r"\s+", " ", (statement or "").strip())
        if len(stmt) < 20:
            return
        if lock and evidence_is_on_brief and not evidence_is_on_brief(stmt, lock):
            return
        key = stmt.lower()[:120]
        if key in seen:
            return
        seen.add(key)

        # Attach URL from source title if missing
        st = (source_title or "").strip()
        if not url and st:
            url = url_by_title.get(re.sub(r"\s+", " ", st.lower())[:80], "")
        # Pull "(Source: …)" from statement
        m = re.search(r"\(Source:\s*([^)]+)\)\s*$", stmt, re.I)
        if m and not st:
            st = m.group(1).strip()
            if not url:
                url = url_by_title.get(re.sub(r"\s+", " ", st.lower())[:80], "")
        # If still no title, try to recover a clean title from the statement line
        if not st:
            st = _title_from_statement(stmt)

        figs = list(_figures_in(stmt))
        years = _years_in(stmt)
        eid = f"E{len(entries) + 1}"
        entries.append(
            {
                "id": eid,
                "kind": kind,
                "statement": stmt[:500],
                "figure": figs[0] if figs else "",
                "figures": figs[:4],
                "year": years[0] if years else "",
                "geography": _guess_geo(stmt),
                "source_title": st[:200],
                "url": url[:500],
                "confidence": round(float(confidence), 2),
            }
        )

    # Incidents first (news cases)
    for raw in incidents or []:
        line = str(raw).strip()
        if line.upper().startswith("NEWS CASE:"):
            line = line.split(":", 1)[-1].strip()
        _add("incident", line, confidence=0.7)

    # Statistics
    for raw in statistics or []:
        line = str(raw).strip()
        if line.upper().startswith("NEWS CASE:"):
            continue  # already as incidents
        conf = 0.75 if _figures_in(line) else 0.55
        _add("statistic", line, confidence=conf)

    # High-authority sources as citeable context (no number required)
    for row in source_rows[:8]:
        title = row["title"]
        url = row["url"]
        if not title:
            continue
        if lock and evidence_is_on_brief and not evidence_is_on_brief(
            f"{title} {url}", lock
        ):
            continue
        key = title.lower()[:80]
        if key in seen:
            continue
        # Only add source rows that look topical (not pure SEO junk already filtered)
        _add(
            "source",
            f"Reference: {title}",
            source_title=title,
            url=url,
            confidence=0.5,
        )

    # Prefer entries with URLs / figures
    entries.sort(
        key=lambda e: (
            1 if e.get("url") else 0,
            1 if e.get("figure") else 0,
            float(e.get("confidence") or 0),
        ),
        reverse=True,
    )
    # Re-id after sort
    out = []
    for i, e in enumerate(entries[:max_entries], start=1):
        e = dict(e)
        e["id"] = f"E{i}"
        out.append(e)

    logger.info(
        "Evidence ledger built | entries=%d | with_url=%d | with_figure=%d",
        len(out),
        sum(1 for e in out if e.get("url")),
        sum(1 for e in out if e.get("figure")),
    )
    return out


def format_ledger_for_writer(ledger: Optional[List[Dict[str, Any]]]) -> str:
    """Prompt block: Writer may only use these facts."""
    rows = list(ledger or [])
    if not rows:
        return (
            "EVIDENCE LEDGER: empty.\n"
            "- Do NOT invent percentages, case counts, survey names, or salary bands.\n"
            "- Write a practical, conservative piece from the brief.\n"
            "- If the brief asks for stats that are missing, say on-brief public figures "
            "were limited — do not pad with off-market numbers."
        )
    lines = [
        "EVIDENCE LEDGER (ONLY authorised facts — every number/case in the draft "
        "must come from an entry below; cite source_title + year when present):",
    ]
    for e in rows:
        fig = e.get("figure") or "—"
        year = e.get("year") or "—"
        geo = e.get("geography") or "—"
        src = e.get("source_title") or "unspecified source"
        kind = e.get("kind") or "fact"
        stmt = e.get("statement") or ""
        lines.append(
            f"- [{e.get('id')}] ({kind}) figure={fig} | year={year} | geo={geo} | "
            f"source={src}\n  {stmt}"
        )
    lines.append(
        "Rules: If a figure you want is not in this ledger, omit it. "
        "For incidents, use reported/alleged/under investigation language."
    )
    return "\n".join(lines)


def _looks_like_title(text: str) -> bool:
    """Reject mid-sentence fragments used as Sources labels (e.g. 'o a gap…')."""
    t = re.sub(r"\s+", " ", (text or "").strip())
    if len(t) < 8:
        return False
    # Truncated prose leftovers
    if t.endswith("…") or t.endswith("..."):
        # Titles can be truncated, but not if they start mid-sentence
        if t[:1].islower():
            return False
    if t[:1].islower():
        return False
    # Single/double-letter word starts: "o a gap", "a the", "to redesign"
    if re.match(r"^[a-z]{1,2}(?:\s+[a-z]{1,3})+\s+[a-z]", t):
        return False
    # Too sentence-like without title cues
    if t.count(".") >= 2 and len(t) > 90 and not re.search(r"\b(20[12]\d)\b", t):
        return False
    return True


def _title_from_statement(stmt: str) -> str:
    """Pull a citeable title out of a statistic/incident line when possible."""
    s = re.sub(r"\s+", " ", (stmt or "").strip())
    if not s:
        return ""
    m = re.search(r"\(Source:\s*([^)]+)\)\s*$", s, re.I)
    if m:
        cand = m.group(1).strip().strip(" \"'")
        if _looks_like_title(cand):
            return cand[:200]
    # "Title — outlet" / "Title | Outlet"
    m = re.match(r"^(.{12,120}?)\s+[—|-]\s+.{3,80}$", s)
    if m and _looks_like_title(m.group(1)):
        return m.group(1).strip()[:200]
    if _looks_like_title(s) and len(s) <= 140 and s.count(".") <= 1:
        return s[:200]
    return ""


def _label_from_url(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return ""
    try:
        from urllib.parse import urlparse, unquote

        p = urlparse(u)
        host = (p.netloc or "").replace("www.", "")
        path = unquote((p.path or "").strip("/").split("/")[-1])
        path = re.sub(r"[-_]+", " ", path)
        path = re.sub(r"\.(html?|php|aspx?)$", "", path, flags=re.I)
        if path and len(path) >= 8 and not path.isdigit():
            label = path[:80].strip()
            if label and not label[:1].islower():
                return label
            return label[:1].upper() + label[1:] if label else host
        return host or ""
    except Exception:
        return ""


def ledger_to_citations(ledger: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Sources footer candidates from ledger (URL preferred)."""
    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for e in ledger or []:
        title = str(e.get("source_title") or "").strip()
        url = str(e.get("url") or "").strip()
        stmt = str(e.get("statement") or "").strip()
        if stmt.lower().startswith("reference:"):
            text = title or stmt.replace("Reference:", "").strip()
        else:
            text = title
            if not _looks_like_title(text):
                text = _title_from_statement(stmt)
            if not _looks_like_title(text):
                text = _label_from_url(url)
        # Never ship mid-sentence statement slices as source names
        if not _looks_like_title(text):
            if url:
                text = _label_from_url(url) or "Source"
            else:
                continue
        if not text and not url:
            continue
        key = (url or text).lower()[:120]
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "text": text,
                "url": url,
                "type": "news" if e.get("kind") == "incident" else "web",
                "formatted": f"{text} {url}".strip(),
            }
        )
    return out[:12]


def claim_audit_issues(
    draft: str,
    ledger: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    """
    Review gate: flag numeric / scare claims not backed by the ledger.

    Generic for all brands — no brand-specific lists.
    """
    issues: List[str] = []
    text = draft or ""
    if not text.strip():
        return issues

    # Ignore Sources / Hashtags footer for claim hunting
    body = re.split(
        r"(?=\n(?:##\s+)?Sources\b|\n(?:##\s+)?References\b|\nHashtags:)",
        text,
        maxsplit=1,
        flags=re.I,
    )[0]

    ledger = list(ledger or [])
    allowed: Set[str] = set()
    for e in ledger:
        for f in e.get("figures") or []:
            allowed.add(_norm_figure(str(f)))
        if e.get("figure"):
            allowed.add(_norm_figure(str(e["figure"])))
        # Also allow bare digits from statement figures
        allowed |= _figures_in(str(e.get("statement") or ""))

    # Soft match: "1,900" vs "1900"
    def _canon(f: str) -> str:
        return re.sub(r"[,\s]", "", _norm_figure(f))

    allowed_canon = {_canon(a) for a in allowed if a}

    draft_figs = _figures_in(body)
    unsupported = []
    for fig in draft_figs:
        c = _canon(fig)
        if not c or len(re.sub(r"\D", "", c)) < 2:
            continue
        # Ignore tiny numbers that are often structural (1–2, section counts)
        digits = re.sub(r"\D", "", c)
        if digits.isdigit() and int(digits) <= 12 and "%" not in fig:
            continue
        if c in allowed_canon or _norm_figure(fig) in allowed:
            continue
        # Partial: ledger has "5 nannies" and draft has "five" — skip word forms
        unsupported.append(fig)

    if unsupported and ledger:
        issues.append(
            "EVIDENCE_UNSUPPORTED: Draft uses figures not in the Evidence Ledger ("
            + ", ".join(f'"{u}"' for u in unsupported[:6])
            + "). Remove them or replace only with ledger-backed stats."
        )
    elif unsupported and not ledger:
        issues.append(
            "EVIDENCE_UNSUPPORTED: Draft contains statistics but the Evidence Ledger "
            "is empty. Remove invented figures; write a practical brief-led piece."
        )

    # Absolute marketing / certainty without support
    if re.search(
        r"(?i)\b(guaranteed|risk[-\s]?free|100\s*%\s*safe|never fail|"
        r"completely eliminate)\b",
        body,
    ):
        issues.append(
            "EVIDENCE_OVERCLAIM: Soften absolute safety/guarantee language. "
            "Use careful, evidence-based wording."
        )

    # Incident tone: if draft asserts abuse as proven fact with a named case city
    # and ledger has incidents, require reported/alleged nearby — soft heuristic
    if any(e.get("kind") == "incident" for e in ledger):
        if re.search(
            r"(?i)\b(abused toddlers|were abused|nannies abused)\b",
            body,
        ) and not re.search(
            r"(?i)\b(alleged|allegedly|reported|according to|police booked|"
            r"under investigation)\b",
            body,
        ):
            issues.append(
                "EVIDENCE_TONE: News cases must use reported/alleged/"
                "under investigation language — do not state unresolved cases as proven fact."
            )

    return issues


def merge_ledger_into_research(
    research_data: Dict[str, Any],
    brief_lock: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Attach evidence_ledger onto research_data dict (non-destructive)."""
    data = dict(research_data or {})
    ledger = build_evidence_ledger(
        documents=data.get("documents") or [],
        sources=data.get("sources") or [],
        statistics=data.get("statistics") or [],
        incidents=data.get("incidents") or [],
        brief_lock=brief_lock or data.get("brief_lock"),
    )
    data["evidence_ledger"] = ledger
    return data
