"""
Research Service
================

Collects research from multiple sources and returns a unified
research package for the Research Agent.

Responsibilities:
- Search the internal Knowledge Base.
- Search external sources.
- Merge all retrieved content.
- Remove duplicate information.
- Return normalized research data.

This service contains no business logic and does not modify
the LangGraph state directly.
"""

import logging
from typing import Dict, List, Tuple

from schemas.research_schema import ResearchData, ResearchDocument, ResearchSource
from tools.tavily_search import TavilySearch
from tools.youtube_search import YouTubeSearch
from tools.reddit_search import RedditSearch
from tools.news_search import NewsSearch

logger = logging.getLogger(__name__)


class ResearchService:
    """Handles research retrieval from all configured sources."""

    def __init__(self):
        pass

    def run(
        self,
        query: str,
        brand_context: Dict,
    ) -> Dict:
        """
        Execute research using all available sources.
        """
        namespace = brand_context.get("namespace", "")

        # Retrieve internal knowledge.
        kb_docs, kb_sources = self._search_kb(
            query=query,
            namespace=namespace,
        )

        # Retrieve external knowledge.
        web_docs, web_sources = self._search_web(query)

        all_docs = kb_docs + web_docs
        all_sources = kb_sources + web_sources

        # Deduplicate sources by URL
        unique_sources = []
        seen_urls = set()
        for src in all_sources:
            if src.url:
                if src.url not in seen_urls:
                    seen_urls.add(src.url)
                    unique_sources.append(src)
            else:
                unique_sources.append(src)

        # Build final validated data structure
        research_data = ResearchData(
            documents=all_docs,
            total_documents=len(all_docs),
            sources=unique_sources,
            statistics=[],
            citations=[],
        )

        logger.info(
            "Research complete | query='%s' | documents=%d | sources=%d",
            query,
            len(all_docs),
            len(unique_sources),
        )

        return research_data.to_state_dict()

    # ------------------------------------------------------------------

    def _search_kb(
        self,
        query: str,
        namespace: str,
    ) -> Tuple[List[ResearchDocument], List[ResearchSource]]:
        """
        Search the brand-specific Knowledge Base.
        """
        # Connect Vector DB here when needed.
        return [], []

    # ------------------------------------------------------------------

    def _search_web(
        self,
        query: str,
    ) -> Tuple[List[ResearchDocument], List[ResearchSource]]:
        """
        Search external research sources.
        """
        documents: List[ResearchDocument] = []
        sources: List[ResearchSource] = []

        # 1. Tavily Search
        try:
            tavily = TavilySearch()
            tavily_results = tavily.search(query)
            for item in tavily_results:
                text_content = item.get("raw_content") or item.get("content") or ""
                if text_content.strip():
                    documents.append(
                        ResearchDocument(
                            text=text_content.strip(),
                            title=item.get("title") or "",
                            url=item.get("url") or "",
                            source_type="web",
                            relevance_score=float(item.get("score") or 0.0),
                            metadata={},
                        )
                    )
                    sources.append(
                        ResearchSource(
                            title=item.get("title") or "",
                            url=item.get("url") or "",
                            source_type="web",
                            snippet=item.get("content") or "",
                        )
                    )
        except Exception as exc:
            logger.error("Tavily search tool error: %s", exc, exc_info=True)

        # 2. YouTube Search
        try:
            youtube = YouTubeSearch()
            youtube_results = youtube.search(query)
            for item in youtube_results:
                transcript = item.get("transcript") or ""
                if transcript.strip():
                    documents.append(
                        ResearchDocument(
                            text=transcript.strip(),
                            title=item.get("title") or "",
                            url=item.get("url") or "",
                            source_type="youtube",
                            relevance_score=1.0,
                            metadata={
                                "channel": item.get("channel") or "",
                                "video_id": item.get("video_id") or "",
                            },
                        )
                    )
                    sources.append(
                        ResearchSource(
                            title=item.get("title") or "",
                            url=item.get("url") or "",
                            source_type="youtube",
                            published_date=item.get("published_at"),
                            author=item.get("channel") or "",
                            snippet=item.get("description") or "",
                        )
                    )
        except Exception as exc:
            logger.error("YouTube search tool error: %s", exc, exc_info=True)

        # 3. Reddit Search
        try:
            reddit = RedditSearch()
            reddit_results = reddit.search(query)
            for item in reddit_results:
                text_content = item.get("content") or ""
                if item.get("top_comments"):
                    text_content += "\n\nTop Comments:\n" + "\n".join(item["top_comments"])
                if text_content.strip():
                    documents.append(
                        ResearchDocument(
                            text=text_content.strip(),
                            title=item.get("title") or "",
                            url=item.get("url") or "",
                            source_type="reddit",
                            relevance_score=0.5,
                            metadata={
                                "subreddit": item.get("subreddit") or "",
                            },
                        )
                    )
                    sources.append(
                        ResearchSource(
                            title=item.get("title") or "",
                            url=item.get("url") or "",
                            source_type="reddit",
                            author=item.get("author") or "",
                            snippet=(item.get("content") or "")[:200],
                        )
                    )
        except Exception as exc:
            logger.error("Reddit search tool error: %s", exc, exc_info=True)

        # 4. News Search
        try:
            news = NewsSearch()
            news_results = news.search(query)
            for item in news_results:
                text_content = item.get("content") or item.get("description") or ""
                if text_content.strip():
                    documents.append(
                        ResearchDocument(
                            text=text_content.strip(),
                            title=item.get("title") or "",
                            url=item.get("url") or "",
                            source_type="news",
                            relevance_score=0.8,
                            metadata={
                                "source_name": item.get("source") or "",
                            },
                        )
                    )
                    sources.append(
                        ResearchSource(
                            title=item.get("title") or "",
                            url=item.get("url") or "",
                            source_type="news",
                            published_date=item.get("published_at"),
                            author=item.get("source") or "",
                            snippet=item.get("description") or "",
                        )
                    )
        except Exception as exc:
            logger.error("News search tool error: %s", exc, exc_info=True)

        return documents, sources