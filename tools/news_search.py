"""
News Search Tool

Responsibilities:
- Search recent news articles via NewsAPI
- Return clean structured data
- Handle API errors gracefully
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List

import requests

from config.settings import settings

logger = logging.getLogger(__name__)

_CLEAN_OPS_RE = re.compile(r"\s*-site:\S+|\bOR\b|\bAND\b|\(|\)|\"", re.I)
_GEO_WORDS = {
    "india",
    "indian",
    "delhi",
    "mumbai",
    "bangalore",
    "bengaluru",
    "hyderabad",
    "chennai",
    "kolkata",
    "pune",
    "noida",
    "gurgaon",
    "gurugram",
    "usa",
    "america",
    "american",
    "uk",
    "britain",
    "british",
    "canada",
    "canadian",
    "australia",
    "australian",
    "uae",
    "dubai",
    "singapore",
}
_STOP = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "about",
    "write",
    "article",
    "how",
    "can",
    "help",
    "into",
    "that",
    "this",
}


class NewsSearch:

    BASE_URL = "https://newsapi.org/v2/everything"

    def __init__(self):
        self.api_key = settings.NEWS_API_KEY

    def search(self, query: str, page_size: int = 5) -> List[Dict]:
        """
        Search recent news related to a topic.

        Args:
            query: Search query
            page_size: Number of articles to fetch

        Returns:
            List of structured news articles
        """
        if not self.api_key or not str(self.api_key).strip():
            logger.warning("[NewsSearch] NEWS_API_KEY is not configured")
            return []

        q = self._clean_query(query)
        if not q:
            return []

        params = {
            "q": q,
            "pageSize": page_size,
            "language": "en",
            "sortBy": "relevancy",
            "apiKey": self.api_key,
        }

        try:
            response = requests.get(
                self.BASE_URL,
                params=params,
                timeout=15,
            )
            data = response.json() if response.content else {}

            if response.status_code >= 400 or data.get("status") == "error":
                logger.warning(
                    "[NewsSearch] API error status=%s code=%s message=%s",
                    response.status_code,
                    data.get("code"),
                    data.get("message"),
                )
                return []

            articles = []
            for article in data.get("articles", []):
                articles.append(
                    {
                        "title": article.get("title"),
                        "description": article.get("description"),
                        "content": article.get("content"),
                        "url": article.get("url"),
                        "source": article.get("source", {}).get("name"),
                        "published_at": article.get("publishedAt"),
                    }
                )

            logger.info(
                "[NewsSearch] query=%r | results=%d",
                q[:100],
                len(articles),
            )
            return articles

        except Exception as e:
            logger.warning("[NewsSearch] Error: %s", e)
            return []

    @classmethod
    def _clean_query(cls, query: str) -> str:
        """
        NewsAPI ANDs every bare word — long research queries often return 0 hits.
        Strip Tavily operators and rewrite long queries as OR + geo AND.
        """
        raw = (query or "").split("|")[0].strip()
        raw = _CLEAN_OPS_RE.sub(" ", raw)
        raw = re.sub(r"\s+", " ", raw).strip()
        tokens = [
            t
            for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]{1,}", raw)
            if t.lower() not in _STOP
        ]
        # de-dupe case-insensitively, preserve order
        seen = set()
        words: List[str] = []
        for t in tokens:
            key = t.lower()
            if key in seen:
                continue
            seen.add(key)
            words.append(t)

        if not words:
            return ""
        if len(words) <= 3:
            return " ".join(words)

        geo = [w for w in words if w.lower() in _GEO_WORDS]
        topic = [w for w in words if w.lower() not in _GEO_WORDS]
        # Prefer concrete topic terms first (avoids noisy OR matches).
        preferred = {
            "nanny",
            "nannies",
            "daycare",
            "childcare",
            "creche",
            "crèche",
            "abuse",
            "assault",
            "safety",
            "screening",
            "caregiver",
            "babysitter",
            "pocso",
            "ncrb",
            "nri",
            "scam",
            "scams",
            "fraud",
            "frauds",
            "property",
            "real-estate",
            "cyber",
        }
        ranked = [w for w in topic if w.lower() in preferred] + [
            w for w in topic if w.lower() not in preferred
        ]
        # Prefer a tight AND of 2–3 core terms (+ geo) over loose 4-way OR.
        core = ranked[:3]
        if not core:
            return " ".join(words[:4])
        if len(core) == 1:
            q = core[0]
        else:
            q = " AND ".join(core[:2])
            if len(core) >= 3:
                q = f"({q}) AND {core[2]}"
        if geo:
            return f"({q}) AND {geo[0]}"
        return q
