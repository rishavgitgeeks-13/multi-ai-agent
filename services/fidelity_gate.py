"""
Generic fidelity gate (all brands)
==================================

One brief-lock contract shared by Manager → Research → Citation → Writer → Review.

Brand YAML only supplies identity (display_name, tone, CTA, content_style).
This module enforces *generic* fidelity rules so we stop patching Kinvo/MPM/etc.

Rules
-----
1. Brief lock: topic tokens + market(s) extracted once from the user brief.
2. Evidence: keep Sources/docs that overlap the lock; drop zero-overlap and
   clear off-market / off-domain junk (not brand-specific lists).
3. Voice: banned cinematic openers, brand spelling = YAML display_name.
4. CTA: intensity still owned by cta_policy (awareness vs hard).
5. Review: hard FAIL issues when lock is violated.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Market taxonomy (generic — not brand-specific)
# ---------------------------------------------------------------------------

# label → (canonical market id, extra tokens that imply this market)
_MARKET_MARKERS: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "india": (
        "india",
        (
            "india",
            "indian",
            "delhi",
            "ncr",
            "gurgaon",
            "gurugram",
            "mumbai",
            "bangalore",
            "bengaluru",
            "hyderabad",
            "chennai",
            "kolkata",
            "pune",
            "noida",
            "rera",
            "aadhaar",
            "aadhar",
            "ncrb",
            "pocso",
        ),
    ),
    "uk": (
        "uk",
        (
            "uk",
            "u.k.",
            "united kingdom",
            "britain",
            "british",
            "london",
            "manchester",
            "birmingham",
            "dbs",
            "ofsted",
        ),
    ),
    "us": (
        "us",
        (
            "united states",
            "u.s.",
            "u.s.a.",
            "usa",
            "america",
            "american",
            "california",
            "new york",
            "texas",
            "florida",
        ),
    ),
    "uae": (
        "uae",
        ("uae", "dubai", "abu dhabi", "emirates"),
    ),
    "pakistan": (
        "pakistan",
        ("pakistan", "pakistani", "lahore", "karachi", "islamabad"),
    ),
    "singapore": (
        "singapore",
        ("singapore", "singaporean"),
    ),
}

# Generic junk: wrong-domain SEO bait that rides shared verbs (trust, hiring, …)
_GENERIC_JUNK = re.compile(
    r"("
    r"cybercrime\.gov|"
    r"wikipedia\.org/wiki/(cybercrime|internet_fraud)|"
    r"\bcyber[\s-]?crime\b|"
    r"\bcyber[\s-]?fraud\b|"
    r"\binternet fraud\b|"
    r"rent[\s-]?a[\s-]?boyfriend|"
    r"powerdmarc|"
    r"tavily research summary|"
    r"\bhiring an electrician\b|"
    r"\bhiring a plumber\b|"
    r"\bhiring a contractor\b|"
    r"\bspotonvision\b|"
    r"\bmost important currency in b2b\b|"
    r"\bb2b\b.{0,60}\btrust\b|"
    r"\btrust\b.{0,60}\bb2b\b|"
    r"can be a game[- ]changer for your"
    r")",
    re.I,
)

# Only treat cyber/fraud Sources as on-topic when the brief asks for them.
_FRAUD_BRIEF = re.compile(
    r"\b(fraud|scam|cyber|phishing|cheat|embezzl)\b",
    re.I,
)

_CINEMATIC_OPENERS = [
    re.compile(
        r"(?i)^(?:Picture|Imagine)\s+(?:a|an|the|your)\s+[^.!?\n]{8,200}[.!?]\s*"
    ),
    re.compile(
        r"(?i)\b(?:Picture|Imagine)\s+(?:a|an|the)\s+"
        r"(?:family|parent|couple|household|morning|scene|weekday|busy|typical)\b"
        r"[^.!?\n]{0,180}[.!?]\s*"
    ),
    re.compile(r"(?i)\bPicture this(?:\s+scenario)?\s*[:.—-]?\s*"),
    re.compile(r"(?i)\bthis scene is familiar\b[^.!?\n]{0,100}[.!]?\s*"),
    re.compile(r"(?i)\bYour family deserves nothing less\.?\s*"),
    re.compile(r"(?i)\bthe stakes are high\.?\s*"),
]

_STOP = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "your",
        "into",
        "about",
        "write",
        "article",
        "guide",
        "how",
        "what",
        "when",
        "why",
        "who",
        "nri",
        "2024",
        "2025",
        "2026",
        "2027",
        "blog",
        "linkedin",
        "email",
        "words",
        "word",
    }
)


@dataclass
class BriefLock:
    """Immutable fidelity contract for one generation run."""

    topic: str = ""
    topic_tokens: List[str] = field(default_factory=list)
    markets: List[str] = field(default_factory=list)  # canonical ids locked by brief
    market_tokens: List[str] = field(default_factory=list)  # surface forms seen
    brand_display_name: str = ""
    brand_namespace: str = ""
    awareness_first: bool = False
    objective: str = ""
    content_type: str = ""
    wants_fraud_angle: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "BriefLock":
        if not isinstance(data, dict):
            return cls()
        return cls(
            topic=str(data.get("topic") or ""),
            topic_tokens=list(data.get("topic_tokens") or []),
            markets=list(data.get("markets") or []),
            market_tokens=list(data.get("market_tokens") or []),
            brand_display_name=str(data.get("brand_display_name") or ""),
            brand_namespace=str(data.get("brand_namespace") or ""),
            awareness_first=bool(data.get("awareness_first")),
            objective=str(data.get("objective") or ""),
            content_type=str(data.get("content_type") or ""),
            wants_fraud_angle=bool(data.get("wants_fraud_angle")),
        )

    @property
    def token_set(self) -> Set[str]:
        return {t.lower() for t in self.topic_tokens if t}


def _tokens(text: str) -> Set[str]:
    return {
        t
        for t in re.findall(r"[a-z0-9]{3,}", (text or "").lower())
        if t not in _STOP
    }


def _detect_markets(text: str) -> Tuple[List[str], List[str]]:
    """Return (canonical market ids, matched surface tokens)."""
    blob = (text or "").lower()
    markets: List[str] = []
    surfaces: List[str] = []
    for _key, (canon, markers) in _MARKET_MARKERS.items():
        hit = False
        for m in markers:
            if re.search(r"(?<!\w)" + re.escape(m) + r"(?!\w)", blob):
                hit = True
                surfaces.append(m)
        if hit and canon not in markets:
            markets.append(canon)
    return markets, surfaces


def build_brief_lock(
    user_input: str = "",
    primary_topic: str = "",
    brand_context: Optional[Dict[str, Any]] = None,
    objective: str = "",
    content_type: str = "",
) -> BriefLock:
    """Build the fidelity contract once (Manager) for the whole run."""
    brand = brand_context if isinstance(brand_context, dict) else {}
    topic = (primary_topic or user_input or "").strip()
    combined = f"{user_input or ''}\n{primary_topic or ''}".strip()
    markets, market_tokens = _detect_markets(combined)
    tokens = sorted(_tokens(combined))

    display = str(
        brand.get("display_name") or brand.get("brand") or ""
    ).strip()
    namespace = str(brand.get("namespace") or "").strip().lower()

    from services.cta_policy import is_awareness_first

    awareness = is_awareness_first(brand)
    # Mode pack overrides brand heuristic when present
    policy = brand.get("mode_policy") if isinstance(brand.get("mode_policy"), dict) else {}
    mode = str(policy.get("mode") or brand.get("content_mode") or "").lower()
    if mode == "awareness":
        awareness = True
    elif mode in ("lead_gen", "seo_page", "authority"):
        awareness = mode == "awareness"
    obj = (
        objective
        or str(brand.get("objective") or "")
        or ""
    ).strip().lower()
    ct = (
        content_type
        or str(brand.get("content_type") or "")
        or ""
    ).strip().lower()

    lock = BriefLock(
        topic=topic[:500],
        topic_tokens=tokens[:80],
        markets=markets,
        market_tokens=sorted(set(market_tokens))[:40],
        brand_display_name=display,
        brand_namespace=namespace,
        awareness_first=awareness,
        objective=obj,
        content_type=ct,
        wants_fraud_angle=bool(_FRAUD_BRIEF.search(combined)),
    )
    logger.info(
        "BriefLock built | markets=%s | tokens=%d | brand=%s | awareness=%s",
        lock.markets,
        len(lock.topic_tokens),
        lock.brand_namespace or "-",
        lock.awareness_first,
    )
    return lock


def writer_prompt_block(lock: BriefLock) -> str:
    """Inject into Writer prompts — generic, brand-agnostic rules."""
    markets = ", ".join(lock.markets) if lock.markets else "none locked (stay neutral)"
    brand = lock.brand_display_name or "the brand"
    lines = [
        "FIDELITY LOCK (mandatory for every brand):",
        f"- PRIMARY TOPIC: {lock.topic or '(see user brief)'}",
        f"- MARKETS LOCKED BY BRIEF: {markets}",
        f"- Brand display name (spell exactly): {brand}",
        "- Do not invent cinematic openers (Picture a… / Imagine a…). Start with the reader problem.",
        "- Cases and stats must match the locked market(s). If research is off-market, omit it — do not pad.",
        "- Sources / claims must serve the brief tokens, not a parallel SEO topic.",
    ]
    if lock.awareness_first:
        lines.append(
            f"- Awareness-first: introduce {brand} only late; CTA once in the conclusion only."
        )
    if not lock.wants_fraud_angle:
        lines.append(
            "- Brief is not a fraud/scam piece — do not inject cyber-fraud stats or unrelated crime dumps."
        )
    return "\n".join(lines)


def research_query_constraints(lock: BriefLock) -> str:
    """Short suffix hints for search queries (no forced wrong geography)."""
    bits: List[str] = []
    if "india" in lock.markets:
        bits.append("India")
    elif "uk" in lock.markets:
        bits.append("United Kingdom")
    elif "us" in lock.markets:
        bits.append("United States")
    elif "uae" in lock.markets:
        bits.append("UAE")
    # Prefer top brief tokens that are not pure stop noise
    for t in lock.topic_tokens[:4]:
        if t not in bits and len(t) > 3:
            bits.append(t)
    return " ".join(bits)[:180]


def plan_research_queries(lock: BriefLock, query: str = "") -> List[str]:
    """
    Generic research fan-out for every brand.

    Builds on-brief search queries from the lock (topic + market + intent),
    not from brand-specific patch lists.
    """
    topic = (lock.topic or query or "").strip()
    if not topic:
        return []
    core = topic.split("|")[0].strip()[:280]
    years = re.findall(r"\b(20[12]\d)\b", core)
    year_bit = (
        " OR ".join(sorted(set(years))[:4]) if years else "2023 OR 2024 OR 2025 OR 2026"
    )
    market_bit = research_query_constraints(lock)
    base = f"{core} {market_bit}".strip() if market_bit else core

    queries: List[str] = [base[:500]]
    ql = core.lower()

    # Intent-aware extras (generic)
    if re.search(
        r"\b(how to|verify|checklist|steps|guide|hiring|hire)\b", ql
    ):
        queries.append(
            f"{base} checklist OR verification OR best practices ({year_bit})"[:500]
        )
    if re.search(r"\b(case|abuse|incident|arrest|scam|fraud|news)\b", ql):
        queries.append(
            f"{base} news cases OR reported OR police ({year_bit})"[:500]
        )
    if re.search(
        r"\b(stat|data|market|yield|salary|cost|price|trend|202[4-6])\b", ql
    ) or True:
        # Always pull one evidence/stats query so planning has numbers
        queries.append(
            f"{base} statistics OR survey OR report OR data ({year_bit})"[:500]
        )

    # Dedupe
    seen = set()
    out: List[str] = []
    for q in queries:
        key = q.lower().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(q)
        if len(out) >= 4:
            break
    return out


def evidence_is_on_brief(text: str, lock: BriefLock) -> bool:
    """Soft check: keep evidence that overlaps topic or locked market."""
    blob = (text or "").lower()
    if not blob:
        return False
    if _GENERIC_JUNK.search(blob):
        return False
    if not lock.wants_fraud_angle and re.search(
        r"\b(cyber[\s-]?fraud|cyber[\s-]?crime)\b", blob
    ):
        return False
    tokens = lock.token_set
    if tokens and sum(1 for t in tokens if t in blob) > 0:
        return True
    markets, _ = _detect_markets(blob)
    if lock.markets and markets and set(lock.markets).isdisjoint(set(markets)):
        return False
    if lock.markets and markets and not set(lock.markets).isdisjoint(set(markets)):
        return True
    # No market lock / no tokens — keep (avoid emptying research)
    if not tokens and not lock.markets:
        return True
    return False


def summarize_research_for_plan(
    research_data: Optional[Dict[str, Any]],
    lock: BriefLock,
    max_stats: int = 6,
    max_incidents: int = 4,
    max_sources: int = 6,
) -> str:
    """Compact evidence block for Strategy outline planning."""
    data = research_data if isinstance(research_data, dict) else {}
    stats = [
        str(s).strip()
        for s in (data.get("statistics") or [])
        if str(s).strip() and evidence_is_on_brief(str(s), lock)
    ][:max_stats]
    incidents = []
    for item in data.get("news_incidents") or data.get("incidents") or []:
        if isinstance(item, dict):
            line = str(
                item.get("summary")
                or item.get("text")
                or item.get("title")
                or ""
            ).strip()
        else:
            line = str(item).strip()
        if line and evidence_is_on_brief(line, lock):
            incidents.append(line)
        if len(incidents) >= max_incidents:
            break
    sources = []
    for src in data.get("sources") or []:
        if isinstance(src, dict):
            title = str(src.get("title") or src.get("name") or "").strip()
            url = str(src.get("url") or "").strip()
            line = f"{title} | {url}".strip(" |")
        else:
            line = str(src).strip()
        if line and evidence_is_on_brief(line, lock):
            sources.append(line)
        if len(sources) >= max_sources:
            break

    parts = []
    if stats:
        parts.append("STATS:\n" + "\n".join(f"- {s}" for s in stats))
    if incidents:
        parts.append("REPORTED CASES:\n" + "\n".join(f"- {i}" for i in incidents))
    if sources:
        parts.append("SOURCES:\n" + "\n".join(f"- {s}" for s in sources))
    if not parts:
        return (
            "RESEARCH FINDINGS: limited on-brief evidence. "
            "Plan practical sections from the brief; do not invent stats."
        )
    return "RESEARCH FINDINGS (plan sections that can use this evidence):\n" + "\n\n".join(
        parts
    )


def _citation_markets(blob: str) -> List[str]:
    markets, _ = _detect_markets(blob)
    return markets


def filter_citations(
    citations: List[Dict[str, Any]],
    lock: BriefLock,
    user_input: str = "",
) -> List[Dict[str, Any]]:
    """
    Generic Sources filter for every brand.

    Keep items with topic overlap; drop junk, duplicates, and clear off-market
    hits when the brief locked one or more markets.
    """
    if not citations:
        return citations

    # Merge lock with any extra topic text
    topic_tokens = set(lock.token_set)
    topic_tokens |= _tokens(user_input or "")
    topic_tokens |= _tokens(lock.topic or "")

    locked_markets = set(lock.markets or [])
    scored: List[Tuple[int, Dict[str, Any]]] = []
    seen_titles: Set[str] = set()

    for cit in citations:
        text = str(cit.get("text") or "")
        url = str(cit.get("url") or "")
        formatted = str(cit.get("formatted") or "")
        blob = f"{text} {url} {formatted}".lower()

        if _GENERIC_JUNK.search(blob):
            continue
        if not lock.wants_fraud_angle and re.search(
            r"\b(cyber[\s-]?fraud|cyber[\s-]?crime|internet fraud)\b", blob
        ):
            continue

        title_key = re.sub(r"\s+", " ", text.strip().lower())[:80]
        if title_key and title_key in seen_titles:
            continue
        if title_key:
            seen_titles.add(title_key)

        cit_markets = set(_citation_markets(blob))
        # Off-market: brief locked market(s), citation clearly elsewhere, no shared market
        if locked_markets and cit_markets and locked_markets.isdisjoint(cit_markets):
            continue

        overlap = sum(1 for t in topic_tokens if t in blob)
        # Soft domain boost from brief market tokens
        for m in lock.market_tokens:
            if m and m in blob:
                overlap += 1
                break
        # On-market news with weak keyword overlap still counts as relevant
        if locked_markets and cit_markets and not locked_markets.isdisjoint(cit_markets):
            overlap = max(overlap, 1)

        scored.append((overlap, cit))

    if not scored:
        logger.info("FidelityGate filtered all citations")
        return []

    scored.sort(key=lambda x: x[0], reverse=True)
    with_overlap = [c for s, c in scored if s > 0]
    without = [c for s, c in scored if s == 0]
    # Prefer overlap; keep a few non-overlapping non-junk as backup only if nothing matches
    ordered = with_overlap if with_overlap else without
    return ordered[:12]


def enforce_draft(draft: str, lock: BriefLock) -> str:
    """
    Deterministic voice / brand fidelity cleanup for every brand.
    """
    text = draft or ""
    if not text:
        return text

    for pat in _CINEMATIC_OPENERS:
        text = pat.sub("", text)

    # Brand spelling: force YAML display_name over glued/mangled variants
    name = (lock.brand_display_name or "").strip()
    if name and " " in name:
        glued = re.sub(r"\s+", "", name)
        # Replace glued form (KinvoCare) with proper display name
        text = re.sub(
            rf"\b{re.escape(glued)}\b",
            name,
            text,
            flags=re.I,
        )
        # Also fix "KinvoCare's" → "Kinvo Care's"
        text = re.sub(
            rf"\b{re.escape(glued)}('s)\b",
            name + r"\1",
            text,
            flags=re.I,
        )

    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def humanize_review_issues(draft: str) -> List[str]:
    """
    Generic human-voice QC for Review (all brands).

    Fails drafts that still read as AI templates: cinematic openers,
    stock transitions, banned contractions, dash-heavy copy, brochure closers.
    """
    issues: List[str] = []
    text = draft or ""
    if not text.strip():
        return issues

    lower = text.lower()
    body = re.split(
        r"(?=\n(?:##\s+)?Sources\b|\n(?:##\s+)?References\b|\nHashtags:)",
        text,
        maxsplit=1,
        flags=re.I,
    )[0]

    # Cinematic / template openers
    if re.search(
        r"(?im)^(?:Picture|Imagine)\s+(?:a|an|the|your)\b",
        body,
    ) or re.search(
        r"(?i)\bPicture this(?:\s+scenario)?\b",
        body,
    ):
        issues.append(
            "HUMANIZE_OPENER: Opens with a cinematic Picture/Imagine scene. "
            "Rewrite the intro in a natural human voice — start with the reader's real situation."
        )

    # Classic AI transitions / fillers (count density)
    tells = [
        "moreover",
        "furthermore",
        "in conclusion",
        "in summary",
        "it's worth noting",
        "it is worth noting",
        "in today's fast-paced",
        "in today's digital",
        "unlock the power",
        "game-changer",
        "game changer",
        "a testament to",
        "in the realm of",
        "in the landscape of",
        "delve into",
        "navigate the",
        "cutting-edge",
        "seamless experience",
        "holistic approach",
        "your family deserves nothing less",
        "the stakes are high",
        "this article explores",
        "in this article, we will",
        "when it comes to",
        "at the end of the day",
        "needless to say",
        "future-ready",
        "future ready",
        "the payoff is clear",
        "the numbers are hard to ignore",
        "changes the game",
        "workforce of the future",
        "what matters now",
        "stay ahead of the curve",
        "end-to-end solution",
        "best-in-class",
        "digital transformation journey",
    ]
    hit_tells = [t for t in tells if t in lower]
    if len(hit_tells) >= 2:
        issues.append(
            "HUMANIZE_CLICHE: AI-cliché phrases still present ("
            + ", ".join(f'"{t}"' for t in hit_tells[:6])
            + "). Rewrite those lines in plain human language."
        )
    elif hit_tells:
        issues.append(
            "HUMANIZE_CLICHE: Remove AI stock phrase "
            f'"{hit_tells[0]}" and rewrite naturally.'
        )

    # Brand style: no contractions in published body
    contraction_hits = re.findall(
        r"\b(?:you're|we're|they're|it's|that's|what's|who's|"
        r"I'm|I've|I'd|I'll|don't|doesn't|didn't|can't|won't|"
        r"shouldn't|wouldn't|couldn't|isn't|aren't|wasn't|weren't|"
        r"hasn't|haven't|hadn't|let's|there's|here's)\b",
        body,
        flags=re.I,
    )
    # Also curly apostrophe forms
    contraction_hits += re.findall(
        r"\b(?:you|we|they|it|that|what|who|I|do|does|did|can|will|"
        r"should|would|could|is|are|was|were|has|have|had|let|there|here)"
        r"['’](?:re|ve|ll|d|s|t)\b",
        body,
        flags=re.I,
    )
    # Dedupe case-insensitive
    uniq = []
    seen = set()
    for c in contraction_hits:
        key = c.lower().replace("’", "'")
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    if len(uniq) >= 3:
        issues.append(
            "HUMANIZE_CONTRACTIONS: Published style bans contractions. "
            f"Expand forms such as {', '.join(uniq[:5])} "
            "(you are / it is / do not / I would)."
        )

    # Em/en dashes in body (URLs excluded roughly)
    body_no_urls = re.sub(r"https?://\S+", " ", body)
    dash_count = len(re.findall(r"[–—−]", body_no_urls))
    # Also many spaced hyphens used as dashes: " word - word "
    spaced_hyphen = len(re.findall(r"\s-\s", body_no_urls))
    if dash_count >= 2 or spaced_hyphen >= 4:
        issues.append(
            "HUMANIZE_DASHES: Draft still uses dash characters. "
            "Rewrite without hyphens/en/em dashes in body copy (URLs excepted)."
        )

    # Brochure / salesy template closer
    if re.search(
        r"(?i)\b(?:your family deserves nothing less|look no further|"
        r"we(?:'|’)ve got you covered|rest assured)\b",
        body,
    ):
        issues.append(
            "HUMANIZE_BROCHURE: Remove brochure closers. End like a human advisor, not a flyer."
        )

    # Robotic section scaffolding: many sections starting the same way
    h2_leads = re.findall(r"(?im)^##\s+.+\n+([A-Z][^\n]{0,40})", body)
    if len(h2_leads) >= 4:
        same = 0
        for lead in h2_leads:
            if re.match(
                r"(?i)^(In today|When it comes|It is important|Moreover|Furthermore)",
                lead,
            ):
                same += 1
        if same >= 2:
            issues.append(
                "HUMANIZE_STRUCTURE: Sections open with the same AI scaffolding. "
                "Vary openings; write like a person explaining to a colleague."
            )

    # Stacked hedges / throat-clearing
    hedge_hits = re.findall(
        r"(?i)\b(it is important to(?: note)?|it is worth noting|"
        r"generally speaking|to some extent|needless to say|as we (?:all )?know|"
        r"in many cases|on the one hand)\b",
        body,
    )
    if len(hedge_hits) >= 4:
        issues.append(
            "HUMANIZE_HEDGE: Too many stacked hedges. Cut throat-clearing and state "
            "the point in plain language."
        )

    # Rhetorical triad abuse
    triads = re.findall(
        r"\b([A-Za-z][A-Za-z\- ]{2,24}),\s+([A-Za-z][A-Za-z\- ]{2,24}),?\s+and\s+"
        r"([A-Za-z][A-Za-z\- ]{2,24})\b",
        body,
    )
    if len(triads) >= 4:
        issues.append(
            "HUMANIZE_TRIAD: Repeated three-part rhetorical stacks feel AI-generated. "
            "Prefer one concrete detail over \"X, Y, and Z\" cadence."
        )

    return issues


def demonstrate_review_issues(
    draft: str,
    content_type: str = "article",
    demo_style: str = "",
) -> List[str]:
    """
    9+ QC: good ideas that only *explain* abstract claims (not demonstrated in a
    real scene) should not clear the premium bar.

    Soft heuristic for long-form — flags when abstract thesis language outweighs
    concrete demonstrations. demo_style (family | business | case_research |
    practical_steps) tunes which demo cues count.
    """
    issues: List[str] = []
    ct = (content_type or "").lower()
    if ct not in ("blog", "article"):
        return issues

    text = draft or ""
    body = re.split(
        r"(?=\n(?:##\s+)?Sources\b|\n(?:##\s+)?References\b|\nHashtags:)",
        text,
        maxsplit=1,
        flags=re.I,
    )[0]
    words = len(body.split())
    if words < 600:
        return issues

    abstract_patterns = [
        r"\bit is not just about\b",
        r"\bthis is not about\b",
        r"\bthe real (?:bottleneck|challenge|key|answer)\b",
        r"\bthat is where .{0,40}changes the game\b",
        r"\bthe numbers are hard to ignore\b",
        r"\bthe payoff is clear\b",
        r"\bwhat matters now\b",
        r"\bthe smartest (?:companies|teams|businesses)\b",
        r"\ba way to\b",
        r"\bit is about (?:changing|reclaiming|building|how)\b",
        r"\bnot just a tech upgrade\b",
        r"\bfuture[- ]ready\b",
        r"\bworkforce of the future\b",
        r"\bseamless(?:ly)?\b",
        r"\brethinking old processes\b",
    ]
    style = (demo_style or "business").strip().lower()
    demo_patterns = [
        r"\bfor example\b",
        r"\bfor instance\b",
        r"\bon a typical (?:day|monday|morning)\b",
        r"\binstead of\b",
    ]
    if style in ("business", "case_research", "practical_steps", ""):
        demo_patterns.extend(
            [
                r"\bCRM\b",
                r"\b(?:inbox|ticket|pipeline|dashboard|spreadsheet)\b",
                r"\bUSD\s*\d",
                r"\b\$\s*\d",
                r"\b\d{1,3}\s*%\b",
                r"\b\d+\s*(?:hours?|days?|minutes?|weeks?)\b",
                r"\b(?:copy(?:ing)? and paste|triage|rout(?:e|es|ing)|reconcil)\b",
                r"\b(?:ops lead|founder|rep|agent|finance|manager)\b.{0,80}\b(?:opens|flags|misses|scores|handles)\b",
            ]
        )
    if style in ("family", "case_research", ""):
        demo_patterns.extend(
            [
                r"\bfor example\b.{0,40}\b(?:parent|caregiver|family|client)\b",
                r"\b(?:hand[- ]?off|drop[- ]?off|pick[- ]?up|check[- ]?in) (?:at|by|around)\b",
                r"\b(?:sick day|backup|night feed|school run|morning routine)\b",
                r"\bwhen the (?:caregiver|provider|specialist)\b",
                r"\bif the (?:caregiver|provider|specialist)\b",
                r"\bbackground check\b",
                r"\breplacement (?:plan|provider|caregiver)\b",
                r"\bat \d{1,2}\s*(?:am|pm)\b",
            ]
        )
    if style == "practical_steps":
        demo_patterns.extend(
            [
                r"\bstep\s*\d\b",
                r"\bfirst[,:]\b",
                r"\bnext[,:]\b",
                r"\bchecklist\b",
                r"\bdo this\b",
            ]
        )
    if style == "case_research":
        demo_patterns.extend(
            [
                r"\baccording to\b",
                r"\bcase study\b",
                r"\bin a (?:202\d|study|survey|report)\b",
                r"\bresearch (?:shows|found|suggests)\b",
            ]
        )

    scene_label = {
        "family": "real family/reader scene",
        "case_research": "case or research scene",
        "practical_steps": "practical step the searcher can use",
        "business": "real business scene",
    }.get(style, "concrete scene")

    lower = body.lower()
    abstract_hits = sum(1 for p in abstract_patterns if re.search(p, lower, re.I))
    demo_hits = sum(1 for p in demo_patterns if re.search(p, lower, re.I))

    # Each H2 body should ideally show something; count H2s with zero demo cues
    sections = re.split(r"(?im)^##\s+", body)
    thin_sections = 0
    checked = 0
    for sec in sections[1:]:
        checked += 1
        sec_l = sec.lower()
        if len(sec_l.split()) < 40:
            continue
        if not any(re.search(p, sec_l, re.I) for p in demo_patterns):
            # Skip conclusion-ish sections
            heading = sec.split("\n", 1)[0].strip().lower()
            if heading.startswith(("conclusion", "next step", "sources")):
                continue
            thin_sections += 1

    # Long articles need enough concrete demos; abstract thesis language alone is an 8.5 ceiling
    if demo_hits < 3 and abstract_hits >= 2:
        issues.append(
            f"DEMONSTRATE_THIN: Strong ideas stay abstract. Cap is ~8.5 until you "
            f"show what each claim looks like in a {scene_label} "
            "(workflow moment, decision trade-off, cost/time impact, or lived scene). "
            "Rewrite 2–3 explanatory passages into concrete demonstrations."
        )
    elif thin_sections >= 2 and checked >= 4:
        issues.append(
            f"DEMONSTRATE_SECTIONS: At least two sections explain without showing. "
            f"In each thin section, add one {scene_label} "
            "(who / tool / failure or win / result) instead of only defining the idea."
        )
    elif abstract_hits >= 5 and demo_hits < abstract_hits:
        issues.append(
            f"DEMONSTRATE_THIN: Too much thesis language vs demonstration. "
            f"Replace 2–3 abstract claims with {scene_label}s "
            "(who, tool, what breaks or improves, measurable result)."
        )

    return issues


def review_fidelity_issues(
    draft: str,
    lock: BriefLock,
    citations: Optional[Iterable[Any]] = None,
) -> List[str]:
    """Rule-based Review issues — generic FAIL reasons."""
    issues: List[str] = []
    text = draft or ""
    lower = text.lower()

    if re.search(
        r"(?i)\b(?:Picture|Imagine)\s+(?:a|an|the|your)\s+(?:family|parent|weekday|busy|morning)\b",
        text,
    ):
        issues.append(
            "FIDELITY_OPENER: Remove cinematic Picture/Imagine openers. "
            "Start with the reader's real situation in plain language."
        )

    name = (lock.brand_display_name or "").strip()
    if name and " " in name:
        glued = re.sub(r"\s+", "", name).lower()
        if glued and glued in re.sub(r"\s+", "", lower):
            # glued appears as one token
            if re.search(rf"\b{re.escape(glued)}\b", lower):
                issues.append(
                    f"FIDELITY_BRAND_NAME: Spell the brand exactly as "
                    f"\"{name}\" (do not glue words together)."
                )

    if lock.awareness_first and name:
        # Hard-sell H2 before conclusion
        if re.search(
            rf"(?im)^##\s+.*\b(book a consultation|next steps:?\s*book)\b",
            text,
        ) and re.search(r"(?im)^##\s+conclusion\b", text):
            issues.append(
                "FIDELITY_CTA: Awareness-first brands must not use a separate "
                "sales 'Book a Consultation' section before Conclusion. "
                "One CTA in the conclusion only."
            )

    # Off-market body dumps when brief locked a market
    if lock.markets:
        locked = set(lock.markets)
        # Strong foreign market markers in body
        body_markets = set(_citation_markets(text))
        foreign = body_markets - locked
        # Ignore pakistan-only scare when india locked, etc.
        if foreign and locked:
            # Only flag if foreign market is emphasized with a stat-like pattern
            for fm in foreign:
                markers = _MARKET_MARKERS.get(fm, (fm, (fm,)))[1]
                for m in markers[:3]:
                    if re.search(
                        rf"(?i)\b{re.escape(m)}\b.{{0,80}}\b("
                        rf"cases?|reported|arrest|abuse|statistics?|percent)\b",
                        text,
                    ):
                        issues.append(
                            "FIDELITY_MARKET: Draft uses off-market "
                            f"'{m}' evidence but the brief locked "
                            f"{', '.join(sorted(locked))}. Remove or replace "
                            "with on-market sources."
                        )
                        break
                else:
                    continue
                break

    # Citation list quality (if provided)
    if citations is not None:
        filtered = filter_citations(
            [
                c
                if isinstance(c, dict)
                else {"text": str(c), "url": "", "formatted": str(c)}
                for c in citations
            ],
            lock,
        )
        raw_n = len(list(citations)) if not isinstance(citations, list) else len(citations)
        # Re-count from original list length
        try:
            raw_list = list(citations)
            raw_n = len(raw_list)
        except Exception:
            raw_n = 0
        if raw_n >= 4 and len(filtered) <= max(1, raw_n // 3):
            issues.append(
                "FIDELITY_SOURCES: Many Sources are off-brief or off-market. "
                "Keep only sources that match the brief lock."
            )

    return issues
