"""
Zero-junk packaging gate (generic — all brands)
===============================================

Deterministic last-mile filters so packaged articles never ship:
  - mid-sentence / mid-word Sources labels ("o a gap…")
  - tool / internal source leaks (Tavily, DuckDuckGo, …)
  - hashtags for verticals never mentioned in the body
  - prompt / workflow meta echoes

Call from JSONBuilder (packaging) and gold eval (hard fail).
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import unquote, urlparse

# Tool / vendor names that must never appear as citations or body attributions
TOOL_LEAK_RE = re.compile(
    r"\b("
    r"tavily|duckduckgo|newsapi|openai|anthropic|langgraph|"
    r"research summary|web search|search summary|internal_synth"
    r")\b",
    re.I,
)

PROMPT_LEAK_RE = re.compile(
    r"(?i)("
    r"ADDITIONAL/?EDITORIAL INTENT|"
    r"PRIMARY TOPIC LOCK|"
    r"AWARENESS-FIRST PACING|"
    r"9\+\s*RULE|"
    r"EVIDENCE LEDGER|"
    r"NEVER paste workflow|"
    r"Return ONLY the|"
    r"CONTENT ANGLE\s*:"
    r")"
)

# Hashtag stem → body cues that must appear somewhere if the tag is kept
_VERTICAL_TAG_CUES: Dict[str, Tuple[str, ...]] = {
    "healthcare": ("health", "hospital", "patient", "clinic", "medical"),
    "healthcareautomation": ("health", "hospital", "patient", "clinic", "medical"),
    "legal": ("legal", "law", "lawyer", "attorney", "compliance"),
    "legalautomation": ("legal", "law", "lawyer", "attorney", "compliance"),
    "fintech": ("fintech", "banking", "payments", "finance"),
    "edtech": ("education", "school", "learning", "student"),
}


def looks_like_title(text: str) -> bool:
    """Reject mid-sentence fragments used as Sources labels."""
    t = re.sub(r"\s+", " ", (text or "").strip())
    if len(t) < 8:
        return False
    if TOOL_LEAK_RE.search(t):
        return False
    if t.endswith("…") or t.endswith("..."):
        if t[:1].islower():
            return False
    if t[:1].islower():
        return False
    if re.match(r"^[a-z]{1,2}(?:\s+[a-z]{1,3})+\s+[a-z]", t):
        return False
    if t.count(".") >= 2 and len(t) > 90 and not re.search(r"\b(20[12]\d)\b", t):
        return False
    # Truncated prose ending mid-word ("… The MIT report fr")
    if re.search(r"\b[a-z]{1,3}$", t) and (t.endswith("…") or t.endswith("...")):
        return False
    return True


def label_from_url(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return ""
    try:
        p = urlparse(u)
        host = (p.netloc or "").replace("www.", "")
        path = unquote((p.path or "").strip("/").split("/")[-1])
        path = re.sub(r"[-_]+", " ", path)
        path = re.sub(r"\.(html?|php|aspx?)$", "", path, flags=re.I)
        if path and len(path) >= 8 and not path.isdigit():
            label = path[:80].strip()
            if label and label[:1].islower():
                label = label[:1].upper() + label[1:]
            return label
        return host or ""
    except Exception:
        return ""


def sanitize_source_label(label: str, url: str = "", fallback_index: int = 1) -> str:
    """
    Return a publishable Sources label, or a safe fallback.

    Never returns mid-sentence junk like 'o a gap between…'.
    """
    raw = re.sub(r"\s+", " ", (label or "").strip())
    # Strip trailing URL if leaked into label
    if url:
        raw = re.sub(
            r"\s*[—\-.]?\s*" + re.escape(url) + r"\s*$",
            "",
            raw,
        ).strip()
    raw = raw.strip(" .\"'")
    if looks_like_title(raw) and not TOOL_LEAK_RE.search(raw):
        return raw[:200]
    from_url = label_from_url(url)
    if from_url and looks_like_title(from_url):
        return from_url[:200]
    if from_url:
        return from_url[:80]
    try:
        host = (urlparse(url).netloc or "").replace("www.", "")
        if host and not TOOL_LEAK_RE.search(host):
            return host
    except Exception:
        pass
    return f"Source {fallback_index}"


def filter_citations(citations: Optional[Sequence[Any]]) -> List[Dict[str, Any]]:
    """Drop tool-leak cites; sanitize labels on the rest."""
    out: List[Dict[str, Any]] = []
    for i, cit in enumerate(citations or [], start=1):
        if isinstance(cit, dict):
            text = str(cit.get("text") or cit.get("formatted") or "").strip()
            url = str(cit.get("url") or "").strip()
            blob = f"{text} {url}"
        else:
            text = str(cit or "").strip()
            url = ""
            blob = text
        if TOOL_LEAK_RE.search(blob):
            continue
        label = sanitize_source_label(text, url=url, fallback_index=i)
        if not label and not url:
            continue
        # Skip entries that would still be junk and have no URL
        if not looks_like_title(label) and not url:
            continue
        row = dict(cit) if isinstance(cit, dict) else {}
        row["text"] = label
        row["url"] = url
        if "formatted" in row or isinstance(cit, dict):
            row["formatted"] = f"{label} {url}".strip()
        out.append(row)
    return out


def _body_only(draft: str) -> str:
    return re.split(
        r"(?=\n(?:##\s+)?Sources\b|\n(?:##\s+)?References\b|\nHashtags:)",
        draft or "",
        maxsplit=1,
        flags=re.I,
    )[0]


def filter_hashtags(
    hashtags: Optional[Sequence[str]],
    draft_or_body: str = "",
) -> List[str]:
    """
    Keep brand/topic tags; drop vertical tags whose cues never appear in body.
    """
    body = _body_only(draft_or_body).lower()
    out: List[str] = []
    seen = set()
    for raw in hashtags or []:
        tag = str(raw or "").strip()
        if not tag:
            continue
        if not tag.startswith("#"):
            tag = "#" + tag
        key = tag.lower()
        if key in seen:
            continue
        stem = re.sub(r"[^a-z0-9]", "", tag.lower())
        cues = _VERTICAL_TAG_CUES.get(stem)
        if cues and body:
            if not any(c in body for c in cues):
                continue
        if TOOL_LEAK_RE.search(tag):
            continue
        seen.add(key)
        out.append(tag)
    return out


def junk_issues(draft: str) -> List[str]:
    """Detect packaging junk still present in markdown (for Final QC / gold)."""
    issues: List[str] = []
    text = draft or ""
    if not text.strip():
        return issues

    if TOOL_LEAK_RE.search(text):
        issues.append(
            "QC_JUNK_TOOL: Draft cites or names an internal research tool "
            "(Tavily/DuckDuckGo/NewsAPI/…). Remove — use real publishers only."
        )
    if PROMPT_LEAK_RE.search(text):
        issues.append(
            "QC_JUNK_PROMPT: Draft contains workflow/prompt meta. "
            "Remove instructional scaffolding from the published copy."
        )

    # Sources section labels
    m = re.search(
        r"(?ims)^##\s*(?:Sources|References)\s*\n(.*?)(?=\n+#+\s|\n+Hashtags:|\Z)",
        text,
    )
    if m:
        block = m.group(1)
        for line in block.splitlines():
            line = line.strip()
            if not line or not re.match(r"^\d+\.", line):
                continue
            # Extract label from "1. Title" or "1. [Title](url)"
            lab = re.sub(r"^\d+\.\s*", "", line)
            md = re.match(r"\[([^\]]+)\]\([^)]+\)", lab)
            lab = (md.group(1) if md else lab).strip()
            if not looks_like_title(lab):
                issues.append(
                    f"QC_JUNK_SOURCE: Sources label looks like a sentence fragment "
                    f"(\"{lab[:60]}\"). Use a real title or host."
                )
                break

    # Hashtag vertical leakage
    hm = re.search(r"(?im)^Hashtags:\s*(.+)$", text)
    if hm:
        tags = re.findall(r"#\w+", hm.group(1))
        body = _body_only(text).lower()
        leaked = []
        for tag in tags:
            stem = re.sub(r"[^a-z0-9]", "", tag.lower())
            cues = _VERTICAL_TAG_CUES.get(stem)
            if cues and not any(c in body for c in cues):
                leaked.append(tag)
        if leaked:
            issues.append(
                "QC_JUNK_HASHTAG: Hashtags for topics not in the article: "
                + ", ".join(leaked[:6])
                + ". Drop unused verticals."
            )

    return issues


def scrub_packaged_markdown(
    markdown: str,
    *,
    hashtags: Optional[Sequence[str]] = None,
) -> Tuple[str, List[str]]:
    """
    Last-mile scrub of a packaged article.

    Returns (clean_markdown, list of scrub notes).
    """
    notes: List[str] = []
    text = markdown or ""

    # Rebuild Sources block labels if present
    def _fix_sources(match: re.Match) -> str:
        header = match.group(1)
        body = match.group(2)
        lines_out = [header]
        n = 0
        for line in body.splitlines():
            raw = line.strip()
            if not raw:
                continue
            mnum = re.match(r"^(\d+)\.\s*(.+)$", raw)
            if not mnum:
                lines_out.append(line)
                continue
            rest = mnum.group(2).strip()
            url = ""
            label = rest
            md = re.match(r"\[([^\]]+)\]\(([^)]+)\)", rest)
            if md:
                label, url = md.group(1).strip(), md.group(2).strip()
            else:
                um = re.search(r"(https?://\S+)\s*$", rest)
                if um:
                    url = um.group(1)
                    label = rest[: um.start()].strip(" —-.")
            clean = sanitize_source_label(label, url=url, fallback_index=n + 1)
            if clean != label:
                notes.append(f"scrubbed_source:{label[:40]}")
            n += 1
            if url and not TOOL_LEAK_RE.search(url):
                lines_out.append(f"{n}. [{clean}]({url})")
            elif not TOOL_LEAK_RE.search(clean):
                lines_out.append(f"{n}. {clean}")
            else:
                notes.append("dropped_tool_source")
                n -= 1
        return "\n".join(lines_out) + ("\n" if lines_out else "")

    text2 = re.sub(
        r"(?ims)(^##\s*(?:Sources|References)\s*\n)(.*?)(?=\n+#+\s|\n+Hashtags:|\Z)",
        _fix_sources,
        text,
        count=1,
    )
    if text2 != text:
        text = text2

    # Hashtags line
    def _fix_tags(match: re.Match) -> str:
        raw_tags = re.findall(r"#\w+", match.group(1))
        kept = filter_hashtags(raw_tags or list(hashtags or []), text)
        if not kept:
            notes.append("dropped_all_hashtags")
            return ""
        if kept != raw_tags:
            notes.append("scrubbed_hashtags")
        return "Hashtags: " + " ".join(kept) + "\n"

    text = re.sub(r"(?im)^Hashtags:\s*(.+)$\n?", _fix_tags, text)

    return text.rstrip() + ("\n" if text.strip() else ""), notes
