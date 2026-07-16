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
import re
import logging
from typing import Dict, List, Tuple

from config.settings import settings
from schemas.research_schema import (
    ResearchData,
    ResearchDocument,
    ResearchSource,
)
from tools.tavily_search import TavilySearch
from tools.youtube_search import YouTubeSearch
#from tools.reddit_search import RedditSearch
from tools.news_search import NewsSearch
#from memory.vector import VectorStore

logger = logging.getLogger(__name__)


class ResearchService:
    """Handles research retrieval from all configured sources."""


    def _get_source_authority(
        self,
        url: str,
    ) -> float:
        """
        Simple authority score.
        """

        if not url:
            return 0.5

        url = url.lower()

        if any(
            x in url
            for x in [
                ".gov",
                ".edu",
                "wikipedia.org",
                "who.int",
                "openai.com",
                "microsoft.com",
                "google.com",
                "anthropic.com",
            ]
        ):
            return 1.0

        if any(
            x in url
            for x in [
                "forbes.com",
                "techcrunch.com",
                "reuters.com",
                "nytimes.com",
                "wsj.com",
                "github.com",
            ]
        ):
            return 0.9

        if "reddit.com" in url:
            return 0.4

        if "youtube.com" in url:
            return 0.5

        return 0.6          
    

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

        logger.info(
            "ResearchService.run() | query='%s' | namespace='%s'",
            query,
            namespace,
        )

        # Retrieve internal knowledge.
        kb_docs = []
        kb_sources = []

        logger.info(
            "KB search complete | docs=%d | sources=%d",
            len(kb_docs),
            len(kb_sources),
        )

        # Retrieve external knowledge.
        web_docs, web_sources = self._search_web(query)

        logger.info(
            "Web search complete | docs=%d | sources=%d",
            len(web_docs),
            len(web_sources),
        )

        all_docs = kb_docs + web_docs
        all_sources = kb_sources + web_sources

        logger.info(
            "Research summary | KB docs=%d | Web docs=%d | Total docs=%d",
            len(kb_docs),
            len(web_docs),
            len(all_docs),
        )

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

        statistics = self._extract_statistics(
            all_docs
        )

        citations = self._extract_citations(
            unique_sources
        )

        research_data = ResearchData(
            documents=all_docs,
            total_documents=len(all_docs),
            sources=unique_sources,
            statistics=statistics,
            citations=citations,
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
    ) -> Tuple[
        List[ResearchDocument],
        List[ResearchSource],
    ]:
        """
        Search the brand-specific Knowledge Base.
        """

        logger.info(
            "KB search invoked | namespace='%s' | query='%s'",
            namespace,
            query,
        )

        documents: List[ResearchDocument] = []
        sources: List[ResearchSource] = []

        try:
            if not namespace:
                logger.warning(
                    "No namespace provided for KB search."
                )
                return documents, sources

            vector_store = VectorStore()

            results = vector_store.similarity_search(
                query=query,
                namespace=namespace,
                top_k=5,
                doc_type="kb",
                score_threshold=0.35,
            )

            logger.info(
                "KB returned %d results",
                len(results),
            )

            for item in results:

                text = item.get("text", "")
                metadata = item.get(
                    "metadata",
                    {},
                )
                score = float(
                    item.get("score", 0.0)
                )

                title = (
                    metadata.get("title")
                    or metadata.get("source")
                    or "Knowledge Base"
                )

                url = (
                    metadata.get("url")
                    or metadata.get("source")
                    or ""
                )

                documents.append(
                    ResearchDocument(
                        text=text,
                        title=title,
                        url=url,
                        source_type="kb",
                        relevance_score=score,
                        metadata=metadata,
                    )
                )

                sources.append(
                    ResearchSource(
                        title=title,
                        url=url,
                        source_type="kb",
                        snippet=text[:200],
                    )
                )

            return documents, sources

        except Exception as exc:
            logger.error(
                "KB search failed: %s",
                exc,
                exc_info=True,
            )
            return documents, sources

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

        logger.info(
            "Starting external research | query='%s'",
            query,
        )

        # ------------------------------------------------------------------
        # 1. Tavily Search
        # ------------------------------------------------------------------
        try:
            logger.info(
                "Running Tavily search | query='%s'",
                query,
            )

            tavily = TavilySearch()
            # basic + no raw_content keeps research under a few seconds
            # instead of minutes of page scraping.
            tavily_results = tavily.search(
                query,
                search_depth="basic",
                include_raw_content=False,
                include_answer=False,
            )

            logger.info(
                "Tavily returned %d results",
                len(tavily_results),
            )

            for item in tavily_results:
                text_content = (
                    item.get("raw_content")
                    or item.get("content")
                    or ""
                )

                if text_content.strip():
                    documents.append(
                        ResearchDocument(
                            text=text_content.strip(),
                            title=item.get("title") or "",
                            url=item.get("url") or "",
                            source_type="web",
                            relevance_score=float(
                                item.get("score") or 0.0
                            ),
                            metadata={
                                "authority": self._get_source_authority(
                                    item.get("url", "")
                                )
                            },
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
            logger.error(
                "Tavily search tool error: %s",
                exc,
                exc_info=True,
            )

        # ------------------------------------------------------------------
        # 2. YouTube Search
        # ------------------------------------------------------------------
        try:
            logger.info(
                "Running YouTube search | query='%s'",
                query,
            )

            youtube = YouTubeSearch()
            youtube_results = youtube.search(query)

            logger.info(
                "YouTube returned %d results",
                len(youtube_results),
            )

            for item in youtube_results:
                text_content = (
                    item.get("transcript")
                    or item.get("description")
                    or ""
                )

                MAX_TRANSCRIPT_CHARS = 4000
                text_content = text_content[:MAX_TRANSCRIPT_CHARS]

                if text_content.strip():
                    documents.append(
                        ResearchDocument(
                            text=text_content.strip(),
                            title=item.get("title") or "",
                            url=item.get("url") or "",
                            source_type="youtube",
                            relevance_score=1.0,
                            metadata={
                                "channel": item.get("channel") or "",
                                "video_id": item.get("video_id") or "",
                                "authority": 0.5,
                            },
                        )
                    )

                    sources.append(
                        ResearchSource(
                            title=item.get("title") or "",
                            url=item.get("url") or "",
                            source_type="youtube",
                            published_date=item.get(
                                "published_at"
                            ),
                            author=item.get("channel") or "",
                            snippet=item.get("description") or "",
                        )
                    )

        except Exception as exc:
            logger.error(
                "YouTube search tool error: %s",
                exc,
                exc_info=True,
            )

        # ------------------------------------------------------------------
        # 3. Reddit Search
        # ------------------------------------------------------------------
        try:
            logger.info(
                "Running Reddit search via Tavily | query='%s'",
                query,
            )

            tavily = TavilySearch()

            # Cheap path: basic depth, snippets only, fewer results.
            reddit_results = tavily.search(
                f"site:reddit.com {query}",
                max_results=min(3, settings.TAVILY_MAX_RESULTS),
                search_depth="basic",
                include_raw_content=False,
                include_answer=False,
            )

            logger.info(
                "Reddit via Tavily returned %d results",
                len(reddit_results),
            )

            for item in reddit_results:

                text_content = (
                    item.get("content")
                    or item.get("raw_content")
                    or ""
                )

                if text_content.strip():

                    documents.append(
                        ResearchDocument(
                            text=text_content.strip(),
                            title=item.get("title") or "",
                            url=item.get("url") or "",
                            source_type="reddit",
                            relevance_score=float(
                                item.get("score") or 0.5
                            ),
                            metadata={
                                "subreddit": item.get("subreddit") or "",
                                "authority": 0.3,
                            },
                        )
                    )

                    sources.append(
                        ResearchSource(
                            title=item.get("title") or "",
                            url=item.get("url") or "",
                            source_type="reddit",
                            snippet=(
                                item.get("content")
                                or ""
                            )[:200],
                        )
                    )

        except Exception as exc:
            logger.error(
                "Reddit search tool error: %s",
                exc,
                exc_info=True,
            )


        # ------------------------------------------------------------------
        # 4. News Search
        # ------------------------------------------------------------------
        try:
            logger.info(
                "Running News search | query='%s'",
                query,
            )

            news = NewsSearch()
            news_results = news.search(query)

            logger.info(
                "News returned %d results",
                len(news_results),
            )

            for item in news_results:       
                title = item.get("title", "")
                text_content = (
                    item.get("content")
                    or item.get("description")
                    or ""
                )

                if not (title + text_content[:200]).isascii():
                    continue

                

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
                                "authority": self._get_source_authority(
                                    item.get("url", "")
                                ),
                            },
                        )
                    )

                    sources.append(
                        ResearchSource(
                            title=item.get("title") or "",
                            url=item.get("url") or "",
                            source_type="news",
                            published_date=item.get(
                                "published_at"
                            ),
                            author=item.get("source") or "",
                            snippet=item.get("description") or "",
                        )
                    )

        except Exception as exc:
            logger.error(
                "News search tool error: %s",
                exc,
                exc_info=True,
            )

        logger.info(
            "External research complete | documents=%d | sources=%d",
            len(documents),
            len(sources),
        )

        return documents, sources
    
    def _extract_statistics(
        self,
        documents: List[ResearchDocument],
    ) -> List[str]:
        """
        Extract simple statistics from research documents.
        """

        statistics = []
        seen = set()

        patterns = [
            r"\d+%",
            r"\$\d+(?:,\d+)*(?:\.\d+)?",
            r"\d+(?:\.\d+)?\s*(?:million|billion|thousand)",
            r"\d+x",
            r"\d+\s*(?:hours|days|weeks|months|years)",
        ]

        for doc in documents:

            text = doc.text[:3000]

            for pattern in patterns:

                matches = re.finditer(
                    pattern,
                    text,
                    re.IGNORECASE,
                )

                for match in matches:

                    start = max(
                        match.start() - 80,
                        0,
                    )

                    end = min(
                        match.end() + 120,
                        len(text),
                    )

                    snippet = text[start:end].strip()
                    # Attach source title so Writer can attribute claims.
                    source_label = (doc.title or "").strip()
                    if source_label and source_label.lower() not in snippet.lower():
                        snippet = f"{snippet} (Source: {source_label})"

                    if (
                        snippet
                        and snippet not in seen
                    ):
                        seen.add(snippet)
                        statistics.append(snippet)

            if len(statistics) >= 10:
                break

        return statistics[:10]
    
    def _extract_citations(
        self,
        sources: List[ResearchSource],
    ) -> List[str]:
        """
        Build formatted citation strings.
        """

        citations = []
        seen = set()

        for source in sources:

            title = (
                source.title
                or ""
            ).strip()

            url = (
                source.url
                or ""
            ).strip()

            if not title:
                continue

            citation = title

            if source.author:
                citation += f" - {source.author}"

            if source.published_date:
                citation += f" ({source.published_date[:10]})"

            if url:
                citation += f" - {url}"

            if citation not in seen:
                seen.add(citation)
                citations.append(citation)

            if len(citations) >= 8:
                break

        return citations
