"""
DuckDuckGo Search Tool

Web search via DuckDuckGo (no API key).
Primary backend: `ddgs` package (stable).
Fallback: DuckDuckGo Lite HTML if the package path fails.
Region follows geography tokens in the user query only.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List
from urllib.parse import parse_qs, unquote, urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_LITE_URL = "https://lite.duckduckgo.com/lite/"
_DEFAULT_MAX = 6
_TIMEOUT_SEC = 12
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)
_CLEAN_OPS_RE = re.compile(r"\s*-site:\S+", re.I)


class DuckDuckGoSearch:
    """DuckDuckGo web search wrapper (no API key)."""

    def search(self, query: str, max_results: int = _DEFAULT_MAX) -> List[Dict]:
        q = self._clean_query(query)
        if not q:
            return []

        region = self._region_param(query)
        limit = max(1, max_results)

        results = self._search_ddgs(q, region=region, max_results=limit)
        if not results:
            results = self._search_lite(q, region=region, max_results=limit)

        logger.info(
            "DuckDuckGoSearch | query=%r | region=%s | results=%d",
            q[:80],
            region,
            len(results),
        )
        return results

    def _search_ddgs(self, query: str, region: str, max_results: int) -> List[Dict]:
        try:
            from ddgs import DDGS
        except ImportError:
            logger.warning(
                "[DuckDuckGoSearch] ddgs package not installed; using HTML fallback"
            )
            return []

        try:
            raw = list(
                DDGS().text(
                    query,
                    region=region,
                    max_results=max_results,
                )
            )
        except Exception as exc:
            logger.warning("[DuckDuckGoSearch] ddgs backend failed: %s", exc)
            return []

        out: List[Dict] = []
        seen = set()
        for item in raw:
            title = (item.get("title") or "").strip()
            url = (item.get("href") or item.get("link") or "").strip()
            snippet = (item.get("body") or item.get("snippet") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            out.append(
                {
                    "title": title,
                    "url": url,
                    "content": snippet,
                    "snippet": snippet,
                    "score": 0.55,
                }
            )
            if len(out) >= max_results:
                break
        return out

    def _search_lite(self, query: str, region: str, max_results: int) -> List[Dict]:
        try:
            resp = requests.post(
                _LITE_URL,
                data={"q": query, "kl": region},
                timeout=_TIMEOUT_SEC,
                headers={
                    "User-Agent": _UA,
                    "Accept": "text/html,application/xhtml+xml",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Referer": _LITE_URL,
                },
            )
            if resp.status_code >= 400 or not resp.text:
                return []
            low = resp.text.lower()
            if "anomaly" in low or "challenge" in low:
                return []
            return self._parse_lite(resp.text, max_results=max_results)
        except Exception as exc:
            logger.warning("[DuckDuckGoSearch] lite fetch failed: %s", exc)
            return []

    @staticmethod
    def _clean_query(query: str) -> str:
        q = (query or "").split("|")[0].strip()
        q = _CLEAN_OPS_RE.sub(" ", q)
        q = re.sub(r"\s+", " ", q).strip()
        return q[:240]

    @staticmethod
    def _region_param(query: str) -> str:
        t = (query or "").lower()
        if re.search(
            r"\b(india|indian|delhi|ncr|gurgaon|gurugram|mumbai|bangalore|"
            r"bengaluru|hyderabad|chennai|kolkata|pune|noida|pocso|ncrb)\b",
            t,
        ):
            return "in-en"
        if re.search(r"\b(united\s+kingdom|u\.k\.|uk|britain|british)\b", t):
            return "uk-en"
        if re.search(r"\b(canada|canadian)\b", t):
            return "ca-en"
        if re.search(r"\b(australia|australian)\b", t):
            return "au-en"
        if re.search(
            r"\b(united\s+states|u\.s\.a\.?|u\.s\.|usa|america|american)\b",
            t,
        ):
            return "us-en"
        if re.search(r"\b(uae|dubai)\b", t):
            return "ae-en"
        if re.search(r"\b(singapore)\b", t):
            return "sg-en"
        return "wt-wt"

    @classmethod
    def _parse_lite(cls, html: str, max_results: int) -> List[Dict]:
        soup = BeautifulSoup(html or "", "html.parser")
        out: List[Dict] = []
        seen = set()
        links = soup.select("a.result-link")
        snippets = soup.select("td.result-snippet")

        for idx, anchor in enumerate(links):
            if len(out) >= max_results:
                break
            title = anchor.get_text(" ", strip=True)
            url = cls._unwrap_url((anchor.get("href") or "").strip())
            if not url or url in seen or not title:
                continue
            snippet = ""
            if idx < len(snippets):
                snippet = snippets[idx].get_text(" ", strip=True)
            seen.add(url)
            out.append(
                {
                    "title": title,
                    "url": url,
                    "content": snippet,
                    "snippet": snippet,
                    "score": 0.55,
                }
            )
        return out

    @staticmethod
    def _unwrap_url(href: str) -> str:
        if not href:
            return ""
        if href.startswith("//"):
            href = "https:" + href
        parsed = urlparse(href)
        if "duckduckgo.com" in (parsed.netloc or "") and (
            parsed.path.startswith("/l/") or "uddg=" in (parsed.query or "")
        ):
            qs = parse_qs(parsed.query)
            target = (qs.get("uddg") or qs.get("u") or [""])[0]
            if target:
                return unquote(target)
        if href.startswith("http"):
            return href
        return ""
