"""
Google News RSS Search Tool

Fetches recent news via Google News public RSS (no API key).
Geography (hl/gl/ceid) follows tokens in the user query only.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote_plus, urlparse

import requests

logger = logging.getLogger(__name__)

_RSS_URL = "https://news.google.com/rss/search"
_DEFAULT_MAX = 8
_TIMEOUT_SEC = 12
_UA = (
    "Mozilla/5.0 (compatible; SEOMultiAgentResearch/1.0; "
    "+https://news.google.com/rss)"
)

# Strip operators that are useful for Tavily but confuse News RSS.
_CLEAN_OPS_RE = re.compile(
    r"\s*-site:\S+|\bOR\b|\bAND\b|\(|\)",
    re.I,
)


class GoogleNewsRSS:
    """Wrapper around Google News RSS search feeds."""

    def search(self, query: str, max_results: int = _DEFAULT_MAX) -> List[Dict]:
        q = self._clean_query(query)
        if not q:
            return []

        hl, gl, ceid = self._geo_params(query)
        url = (
            f"{_RSS_URL}?q={quote_plus(q)}"
            f"&hl={quote_plus(hl)}&gl={quote_plus(gl)}&ceid={quote_plus(ceid)}"
        )

        try:
            resp = requests.get(
                url,
                timeout=_TIMEOUT_SEC,
                headers={"User-Agent": _UA, "Accept": "application/rss+xml, application/xml, text/xml, */*"},
            )
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("[GoogleNewsRSS] fetch failed: %s", exc)
            return []

        try:
            root = ET.fromstring(resp.content)
        except ET.ParseError as exc:
            logger.warning("[GoogleNewsRSS] XML parse failed: %s", exc)
            return []

        channel = root.find("channel")
        items = channel.findall("item") if channel is not None else root.findall("item")
        articles: List[Dict] = []

        for item in items[: max(1, max_results)]:
            title = self._text(item.find("title"))
            link = self._text(item.find("link"))
            pub = self._text(item.find("pubDate"))
            description = self._text(item.find("description"))
            source_el = item.find("source")
            source_name = self._text(source_el) if source_el is not None else ""
            source_url = ""
            if source_el is not None:
                source_url = (source_el.attrib.get("url") or "").strip()

            # Prefer publisher URL from description HTML when present.
            article_url = self._extract_article_url(description) or link
            snippet = self._strip_html(description)

            if not (title or snippet):
                continue

            articles.append(
                {
                    "title": title,
                    "description": snippet,
                    "content": snippet,
                    "url": article_url,
                    "source": source_name or self._host_label(article_url),
                    "source_url": source_url,
                    "published_at": pub,
                }
            )

        logger.info(
            "GoogleNewsRSS | query=%r | geo=%s/%s | results=%d",
            q[:80],
            gl,
            hl,
            len(articles),
        )
        return articles

    @staticmethod
    def _clean_query(query: str) -> str:
        q = (query or "").split("|")[0].strip()
        q = _CLEAN_OPS_RE.sub(" ", q)
        q = re.sub(r"\s+", " ", q).strip()
        return q[:240]

    @staticmethod
    def _geo_params(query: str) -> Tuple[str, str, str]:
        """Map user geography tokens → Google News hl/gl/ceid."""
        t = (query or "").lower()
        if re.search(
            r"\b(india|indian|delhi|ncr|gurgaon|gurugram|mumbai|bangalore|"
            r"bengaluru|hyderabad|chennai|kolkata|pune|noida|pocso|ncrb)\b",
            t,
        ):
            return "en-IN", "IN", "IN:en"
        if re.search(r"\b(united\s+kingdom|u\.k\.|uk|britain|british)\b", t):
            return "en-GB", "GB", "GB:en"
        if re.search(r"\b(canada|canadian)\b", t):
            return "en-CA", "CA", "CA:en"
        if re.search(r"\b(australia|australian)\b", t):
            return "en-AU", "AU", "AU:en"
        if re.search(
            r"\b(united\s+states|u\.s\.a\.?|u\.s\.|usa|america|american)\b",
            t,
        ):
            return "en-US", "US", "US:en"
        if re.search(r"\b(uae|dubai)\b", t):
            return "en-AE", "AE", "AE:en"
        if re.search(r"\b(singapore)\b", t):
            return "en-SG", "SG", "SG:en"
        # Neutral English default when the user named no market.
        return "en", "US", "US:en"

    @staticmethod
    def _text(el: Optional[ET.Element]) -> str:
        if el is None or el.text is None:
            return ""
        return el.text.strip()

    @staticmethod
    def _strip_html(html: str) -> str:
        text = re.sub(r"<[^>]+>", " ", html or "")
        text = re.sub(r"&nbsp;", " ", text, flags=re.I)
        text = re.sub(r"&amp;", "&", text, flags=re.I)
        text = re.sub(r"&lt;", "<", text, flags=re.I)
        text = re.sub(r"&gt;", ">", text, flags=re.I)
        text = re.sub(r"&quot;", '"', text, flags=re.I)
        text = re.sub(r"&#39;", "'", text)
        return re.sub(r"\s+", " ", text).strip()

    @classmethod
    def _extract_article_url(cls, description_html: str) -> str:
        """Pull the first http(s) href from the RSS description HTML."""
        match = re.search(
            r'href=["\'](https?://[^"\']+)["\']',
            description_html or "",
            re.I,
        )
        if not match:
            return ""
        url = match.group(1).strip()
        # Keep Google article links if that is all we have; prefer non-google hosts.
        host = urlparse(url).netloc.lower()
        if "news.google." in host:
            return ""
        return url

    @staticmethod
    def _host_label(url: str) -> str:
        host = urlparse(url or "").netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return host or "Google News"
