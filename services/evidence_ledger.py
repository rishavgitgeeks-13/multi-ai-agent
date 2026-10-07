"""
Evidence ledger (generic — all brands)
======================================

Research → structured claim pack → Writer may only use these facts.
Sources footer prefers ledger URLs. Review fails unsupported numbers.

Each entry:
  id | kind | claim | statement | figure | sample | year | geography |
  context | source_title | url | confidence
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set

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
_SAMPLE_RE = re.compile(
    r"(?i)\b("
    r"(?:n\s*=\s*\d[\d,]*)|"
    r"(?:sample\s+(?:of\s+)?\d[\d,]*)|"
    r"(?:\d[\d,]*\s+(?:respondents?|adults?|parents?|families|households|"
    r"companies|SMBs?|workers?|employees?|nannies|caregivers?|surveyed))"
    r")\b"
)
_CONTEXT_CUES = re.compile(
    r"(?i)\b(survey|study|report|poll|census|among|of\s+(?:indian|uk|us|nri)|"
    r"in\s+(?:india|delhi|gurgaon|london|202[0-9])|"
    r"according to|as per|findings? from)\b"
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


def _extract_sample(text: str) -> str:
    m = _SAMPLE_RE.search(text or "")
    return (m.group(0).strip() if m else "")[:120]


def _extract_context(text: str) -> str:
    """Short claim context window around the first figure / cue."""
    raw = re.sub(r"\s+", " ", (text or "").strip())
    if not raw:
        return ""
    # Prefer sentence containing the figure
    figs = list(_FIGURE_RE.finditer(raw))
    if figs:
        pos = figs[0].start()
        start = max(0, raw.rfind(".", 0, pos) + 1)
        end = raw.find(".", pos)
        if end < 0:
            end = min(len(raw), pos + 160)
        else:
            end = min(len(raw), end + 1)
        window = raw[start:end].strip()
        if len(window) >= 24:
            return window[:220]
    if _CONTEXT_CUES.search(raw):
        return raw[:220]
    return raw[:180]


def _extract_claim(text: str) -> str:
    """Claim = statement without trailing (Source: …) attribution."""
    stmt = re.sub(r"\s+", " ", (text or "").strip())
    stmt = re.sub(r"\s*\(Source:\s*[^)]+\)\s*$", "", stmt, flags=re.I).strip()
    return stmt[:280]


def _stat_is_attributable(entry: Dict[str, Any]) -> bool:
    """Statistic must have year and/or a named source — orphan numbers are dropped."""
    if (entry.get("kind") or "") != "statistic":
        return True
    has_year = bool(str(entry.get("year") or "").strip())
    has_src = bool(str(entry.get("source_title") or "").strip())
    has_url = bool(str(entry.get("url") or "").strip())
    return has_year or has_src or has_url


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
        claim = _extract_claim(stmt)
        sample = _extract_sample(stmt)
        context = _extract_context(stmt)
        # Prefer source-attributed stats; demote orphan figures later
        if kind == "statistic":
            if years and st:
                confidence = max(float(confidence), 0.85)
            elif years or st or url:
                confidence = max(float(confidence), 0.7)
            else:
                confidence = min(float(confidence), 0.45)
        eid = f"E{len(entries) + 1}"
        entries.append(
            {
                "id": eid,
                "kind": kind,
                "claim": claim,
                "statement": stmt[:500],
                "figure": figs[0] if figs else "",
                "figures": figs[:4],
                "sample": sample,
                "year": years[0] if years else "",
                "geography": _guess_geo(stmt),
                "context": context,
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

    # Drop orphan statistics (no year, source, or URL) when better options exist
    attributed_stats = [
        e for e in entries
        if e.get("kind") != "statistic" or _stat_is_attributable(e)
    ]
    if sum(1 for e in attributed_stats if e.get("kind") == "statistic") >= 1:
        entries = attributed_stats
    else:
        # Keep weak stats only if nothing better — still prefer figures with any cue
        entries = [
            e for e in entries
            if e.get("kind") != "statistic"
            or e.get("figure")
            or e.get("source_title")
        ]

    # Prefer year + source + URL + figure
    entries.sort(
        key=lambda e: (
            1 if e.get("year") else 0,
            1 if e.get("source_title") else 0,
            1 if e.get("url") else 0,
            1 if e.get("figure") else 0,
            1 if e.get("sample") else 0,
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
        "Evidence ledger built | entries=%d | with_url=%d | with_figure=%d | with_year=%d",
        len(out),
        sum(1 for e in out if e.get("url")),
        sum(1 for e in out if e.get("figure")),
        sum(1 for e in out if e.get("year")),
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
        "must come from an entry below):",
        "ATTRIBUTION TEMPLATE when using a statistic:",
        '  According to {source_title} ({year}), {claim} — {sample/context if present}.',
        "Never paraphrase a figure without keeping claim, year, and source. "
        "Do not invent sample size or year if the ledger leaves them blank.",
    ]
    for e in rows:
        fig = e.get("figure") or "—"
        year = e.get("year") or "—"
        geo = e.get("geography") or "—"
        src = e.get("source_title") or "unspecified source"
        sample = e.get("sample") or "—"
        kind = e.get("kind") or "fact"
        claim = e.get("claim") or e.get("statement") or ""
        context = e.get("context") or ""
        lines.append(
            f"- [{e.get('id')}] ({kind}) figure={fig} | year={year} | sample={sample} | "
            f"geo={geo} | source={src}\n"
            f"  claim: {claim}\n"
            f"  context: {context or claim}"
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

    issues.extend(stat_fidelity_issues(body, ledger))
    return issues


def stat_fidelity_issues(
    draft: str,
    ledger: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    """
    Flag paraphrased stats that drop year / source / sample when the ledger had them.

    For each draft figure that matches a ledger entry, require nearby attribution
    to retain year and source_title when those fields were present on the entry.
    """
    issues: List[str] = []
    body = draft or ""
    if not body.strip() or not ledger:
        return issues

    # Build figure → best ledger entry
    by_fig: Dict[str, Dict[str, Any]] = {}
    for e in ledger:
        if (e.get("kind") or "") not in ("statistic", "incident", ""):
            continue
        figs = list(e.get("figures") or [])
        if e.get("figure"):
            figs = [str(e["figure"])] + figs
        for f in figs:
            key = re.sub(r"[,\s]", "", _norm_figure(str(f)))
            if key and key not in by_fig:
                by_fig[key] = e

    if not by_fig:
        return issues

    missing_year = 0
    missing_source = 0
    missing_sample = 0
    checked = 0

    for m in _FIGURE_RE.finditer(body):
        fig = m.group(0)
        digits = re.sub(r"\D", "", fig)
        if digits.isdigit() and int(digits) <= 12 and "%" not in fig:
            continue
        key = re.sub(r"[,\s]", "", _norm_figure(fig))
        entry = by_fig.get(key)
        if not entry:
            continue
        checked += 1
        # Window around the figure (±180 chars)
        start = max(0, m.start() - 180)
        end = min(len(body), m.end() + 180)
        window = body[start:end]
        year = str(entry.get("year") or "").strip()
        src = str(entry.get("source_title") or "").strip()
        sample = str(entry.get("sample") or "").strip()

        if year and year not in window:
            # Also accept "according to …" + any 20xx nearby as soft year
            if not _YEAR_RE.search(window):
                missing_year += 1
        if src:
            # Source match: significant token from title, or "according to"/"as per"
            src_toks = [
                t for t in re.findall(r"[A-Za-z]{4,}", src.lower())
                if t not in {"report", "study", "survey", "news", "article", "the"}
            ][:4]
            has_attr = bool(
                re.search(r"(?i)\b(according to|as per|reports? that)\b", window)
            )
            has_src = any(t in window.lower() for t in src_toks) if src_toks else has_attr
            if not has_src and not has_attr:
                missing_source += 1
        if sample:
            # If ledger had a sample size, prefer keeping a number+respondents cue
            sample_digits = re.sub(r"\D", "", sample)
            if sample_digits and sample_digits not in re.sub(r"\D", "", window):
                if not re.search(
                    r"(?i)\b(respondents?|surveyed|sample|households|parents)\b",
                    window,
                ):
                    missing_sample += 1

    if checked == 0:
        return issues
    if missing_year >= 1:
        issues.append(
            "STAT_FIDELITY_YEAR: A ledger-backed statistic is missing its year in the "
            "nearby sentence. Keep the year when paraphrasing (e.g. 'According to X (2024)…')."
        )
    if missing_source >= 1:
        issues.append(
            "STAT_FIDELITY_SOURCE: A ledger-backed statistic is missing its source name "
            "nearby. Attribute the publisher/report when paraphrasing — do not orphan the figure."
        )
    if missing_sample >= 2:
        issues.append(
            "STAT_FIDELITY_SAMPLE: Sample/cohort context from the ledger was dropped. "
            "Preserve sample size or cohort when the source provided it."
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
