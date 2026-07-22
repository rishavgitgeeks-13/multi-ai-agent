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
from tools.google_news_rss import GoogleNewsRSS
from tools.duckduckgo_search import DuckDuckGoSearch
from tools.news_search import NewsSearch  # optional fallback when NEWS_API_KEY set
#from memory.vector import VectorStore

logger = logging.getLogger(__name__)


class ResearchService:
    """Handles research retrieval from all configured sources."""


    def _get_source_authority(
        self,
        url: str,
    ) -> float:
        """
        Authority score used to rank stats/citations (higher = preferred).
        """

        if not url:
            return 0.4

        url = url.lower()

        # Social / UGC — too weak for stats sections
        if any(
            x in url
            for x in [
                "facebook.com",
                "fb.com",
                "instagram.com",
                "tiktok.com",
                "twitter.com",
                "x.com",
                "reddit.com",
                "quora.com",
                "medium.com",
                "blogspot.",
                "wordpress.com",
                "tumblr.com",
            ]
        ):
            return 0.15

        if any(
            x in url
            for x in [
                ".gov",
                ".gov.in",
                ".edu",
                "pib.gov.in",
                "rbi.org.in",
                "cert-in",
                "ncrb.gov.in",
                "india.gov.in",
                "mea.gov.in",
                "mha.gov.in",
                "unicef.org",
                "who.int",
                "worldbank.org",
                "imf.org",
                "oecd.org",
            ]
        ):
            return 1.0

        if any(
            x in url
            for x in [
                "thehindu.com",
                "indianexpress.com",
                "hindustantimes.com",
                "timesofindia",
                "livemint.com",
                "business-standard.com",
                "reuters.com",
                "bbc.com",
                "bbc.co.uk",
                "nytimes.com",
                "wsj.com",
                "economist.com",
                "forbes.com",
                "techcrunch.com",
                "wikipedia.org",
            ]
        ):
            return 0.95

        if "youtube.com" in url or "youtu.be" in url:
            return 0.45

        return 0.55

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

        search_query = self._enrich_query_for_market(query, brand_context)

        # Retrieve internal knowledge.
        kb_docs = []
        kb_sources = []

        logger.info(
            "KB search complete | docs=%d | sources=%d",
            len(kb_docs),
            len(kb_sources),
        )

        # Retrieve external knowledge.
        web_docs, web_sources = self._search_web(search_query)

        logger.info(
            "Web search complete | docs=%d | sources=%d",
            len(web_docs),
            len(web_sources),
        )

        all_docs = kb_docs + web_docs
        all_sources = kb_sources + web_sources

        # Prefer authoritative sources when extracting stats/citations
        all_docs = sorted(
            all_docs,
            key=lambda d: float((d.metadata or {}).get("authority") or 0.0),
            reverse=True,
        )

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

        unique_sources = sorted(
            unique_sources,
            key=lambda s: self._get_source_authority(s.url or ""),
            reverse=True,
        )

        statistics = self._extract_statistics(all_docs, query=query)
        citations = self._extract_citations(unique_sources)

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

    @staticmethod
    def _enrich_query_for_market(query: str, brand_context: Dict) -> str:
        """
        Bias search toward geography named in the USER query only.
        Brand never forces a market (Kinvo/MPM do not imply India).
        """
        del brand_context  # Reserved for future brand filters; geo is user-driven.
        q = (query or "").strip()
        if not q:
            return q

        # Strip framing pipes so search engines get a clean topical query.
        search_core = q.split("|")[0].strip() or q

        wants_india = bool(
            re.search(
                r"\b(india|indian|delhi|ncr|gurgaon|gurugram|mumbai|bangalore|"
                r"bengaluru|hyderabad|chennai|kolkata|pune|noida|pocso|ncrb)\b",
                search_core,
                re.I,
            )
        )
        wants_us = bool(
            re.search(
                r"\b(united\s+states|u\.s\.a\.?|u\.s\.|usa|america|american)\b",
                search_core,
                re.I,
            )
        )
        wants_uk = bool(
            re.search(
                r"\b(united\s+kingdom|u\.k\.|uk|britain|british)\b",
                search_core,
                re.I,
            )
        )

        if wants_india:
            if not re.search(r"\bindia\b", search_core, re.I):
                search_core = f"{search_core} India"
            # Prefer India institutional / news sources over forum noise.
            if re.search(
                r"\b(nanny|childcare|caregiver|abus\w*|child|pocso|rape|assault)\b",
                search_core,
                re.I,
            ):
                search_core = (
                    f"{search_core} NCRB OR POCSO OR childcare safety India "
                    f"-site:reddit.com -site:facebook.com"
                )
            elif re.search(
                r"\b(nri|non[-\s]?resident|property\s+fraud|real\s+estate\s+scam|"
                r"cyber\s*fraud|scam)\b",
                search_core,
                re.I,
            ):
                search_core = (
                    f"{search_core} NRI property fraud OR cyber fraud India "
                    f"statistics -site:reddit.com -site:facebook.com"
                )
            elif not re.search(r"\bsite:", search_core, re.I):
                search_core = (
                    f"{search_core} India -site:reddit.com -site:facebook.com"
                )
        elif wants_us and not re.search(r"\bsite:", search_core, re.I):
            if not re.search(r"\b(united\s+states|usa|u\.s\.)\b", search_core, re.I):
                search_core = f"{search_core} United States"
            search_core = f"{search_core} -site:reddit.com"
        elif wants_uk and not re.search(r"\bsite:", search_core, re.I):
            if not re.search(r"\b(united\s+kingdom|uk|britain)\b", search_core, re.I):
                search_core = f"{search_core} United Kingdom"
            search_core = f"{search_core} -site:reddit.com"

        return search_core[:500]

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
        # 1b. DuckDuckGo (free web complement — no API key; region from query)
        # ------------------------------------------------------------------
        try:
            logger.info(
                "Running DuckDuckGo search | query='%s'",
                query,
            )
            ddg = DuckDuckGoSearch()
            ddg_results = ddg.search(
                query,
                max_results=getattr(settings, "DUCKDUCKGO_MAX_RESULTS", 6),
            )
            logger.info(
                "DuckDuckGo returned %d results",
                len(ddg_results),
            )
            for item in ddg_results:
                text_content = (
                    item.get("content")
                    or item.get("snippet")
                    or item.get("title")
                    or ""
                ).strip()
                if not text_content:
                    continue
                title = (item.get("title") or "").strip()
                body = text_content
                if title and title not in body:
                    body = f"{title}. {body}"
                documents.append(
                    ResearchDocument(
                        text=body,
                        title=title,
                        url=item.get("url") or "",
                        source_type="web",
                        relevance_score=float(item.get("score") or 0.55),
                        metadata={
                            "provider": "duckduckgo",
                            "authority": self._get_source_authority(
                                item.get("url", "")
                            ),
                        },
                    )
                )
                sources.append(
                    ResearchSource(
                        title=title,
                        url=item.get("url") or "",
                        source_type="web",
                        snippet=(item.get("snippet") or item.get("content") or "")[
                            :300
                        ],
                    )
                )
        except Exception as exc:
            logger.error(
                "DuckDuckGo search tool error: %s",
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
                transcript = (item.get("transcript") or "").strip()
                # Empty transcripts must not become research bodies via thin descriptions.
                if len(transcript) < 80:
                    logger.info(
                        "Skipping YouTube without usable transcript | id=%s",
                        item.get("video_id"),
                    )
                    continue

                text_content = transcript[:4000]
                documents.append(
                    ResearchDocument(
                        text=text_content,
                        title=item.get("title") or "",
                        url=item.get("url") or "",
                        source_type="youtube",
                        relevance_score=0.7,
                        metadata={
                            "channel": item.get("channel") or "",
                            "video_id": item.get("video_id") or "",
                            "authority": 0.45,
                            "has_transcript": True,
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
                        snippet=(item.get("description") or "")[:200],
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
        # 4. Google News RSS (primary news — no API key; geo from user query)
        # ------------------------------------------------------------------
        try:
            logger.info(
                "Running Google News RSS search | query='%s'",
                query,
            )

            gnews = GoogleNewsRSS()
            news_results = gnews.search(
                query,
                max_results=getattr(settings, "GOOGLE_NEWS_MAX_RESULTS", 8),
            )

            logger.info(
                "Google News RSS returned %d results",
                len(news_results),
            )

            for item in news_results:
                title = (item.get("title") or "").strip()
                text_content = (
                    item.get("content")
                    or item.get("description")
                    or title
                    or ""
                ).strip()
                if not text_content:
                    continue

                # Prefer title + snippet so Writer/stats extractors get context.
                body = text_content
                if title and title not in body:
                    body = f"{title}. {body}"

                documents.append(
                    ResearchDocument(
                        text=body,
                        title=title,
                        url=item.get("url") or "",
                        source_type="news",
                        relevance_score=0.85,
                        metadata={
                            "source_name": item.get("source") or "",
                            "provider": "google_news_rss",
                            "authority": self._get_source_authority(
                                item.get("url", "")
                            ),
                        },
                    )
                )

                sources.append(
                    ResearchSource(
                        title=title,
                        url=item.get("url") or "",
                        source_type="news",
                        published_date=item.get("published_at"),
                        author=item.get("source") or "",
                        snippet=(item.get("description") or "")[:300],
                    )
                )

        except Exception as exc:
            logger.error(
                "Google News RSS search error: %s",
                exc,
                exc_info=True,
            )

        # ------------------------------------------------------------------
        # 4b. NewsAPI (optional supplement when key is configured)
        # ------------------------------------------------------------------
        if settings.NEWS_API_KEY:
            try:
                logger.info(
                    "Running NewsAPI supplement | query='%s'",
                    query,
                )
                news = NewsSearch()
                api_results = news.search(
                    query,
                    page_size=settings.NEWS_PAGE_SIZE,
                )
                for item in api_results:
                    text_content = (
                        item.get("content")
                        or item.get("description")
                        or ""
                    ).strip()
                    if not text_content:
                        continue
                    documents.append(
                        ResearchDocument(
                            text=text_content,
                            title=item.get("title") or "",
                            url=item.get("url") or "",
                            source_type="news",
                            relevance_score=0.8,
                            metadata={
                                "source_name": item.get("source") or "",
                                "provider": "newsapi",
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
                            published_date=item.get("published_at"),
                            author=item.get("source") or "",
                            snippet=item.get("description") or "",
                        )
                    )
            except Exception as exc:
                logger.error(
                    "NewsAPI supplement error: %s",
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
        query: str = "",
    ) -> List[str]:
        """
        Extract statistics preferring high-authority, query-relevant snippets.
        Demotes Facebook/social and off-audience filler.
        """

        patterns = [
            r"\d+%",
            r"(?:INR|Rs\.?|₹)\s*\d+(?:,\d+)*(?:\.\d+)?\s*(?:crore|lakh)?",
            r"\$\d+(?:,\d+)*(?:\.\d+)?",
            r"\d+(?:\.\d+)?\s*(?:million|billion|thousand|crore|lakh)",
            r"\d+x",
            r"\d{1,3}(?:,\d{3})+\s*(?:cases|complaints|incidents|frauds?)?",
            r"\d+\s*(?:hours|days|weeks|months|years|cases|complaints)",
        ]

        query_l = (query or "").lower()
        years = set(re.findall(r"\b(20[12]\d)\b", query_l))
        wants_nri = bool(re.search(r"\bnri\b|non[-\s]?resident", query_l))
        wants_property = bool(
            re.search(r"\b(property|real\s*estate|rental|land)\b", query_l)
        )

        scored: List[tuple] = []
        seen = set()

        for doc in documents:
            auth = float((doc.metadata or {}).get("authority") or 0.5)
            url = (doc.url or "").lower()
            # Hard-skip social for stats pool
            if auth < 0.3 or any(
                x in url
                for x in ("facebook.com", "instagram.com", "tiktok.com", "reddit.com")
            ):
                continue

            text = doc.text[:3500]
            title_l = (doc.title or "").lower()

            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    start = max(match.start() - 100, 0)
                    end = min(match.end() + 140, len(text))
                    snippet = text[start:end].strip()
                    source_label = (doc.title or "").strip()
                    if source_label and source_label.lower() not in snippet.lower():
                        snippet = f"{snippet} (Source: {source_label})"
                    if not snippet or snippet in seen:
                        continue
                    seen.add(snippet)

                    snip_l = snippet.lower()
                    score = auth
                    if years and any(y in snip_l or y in title_l for y in years):
                        score += 0.35
                    if wants_nri and re.search(r"\bnri\b|non[-\s]?resident", snip_l):
                        score += 0.4
                    elif wants_nri and re.search(
                        r"\b(adults? in india|indian adults?|three out of four)\b",
                        snip_l,
                    ):
                        # General-population scam stats — keep but demote
                        score -= 0.35
                    if wants_property and re.search(
                        r"\b(property|real\s*estate|land|rental|title)\b", snip_l
                    ):
                        score += 0.25
                    scored.append((score, snippet))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:10]]

    def _extract_citations(
        self,
        sources: List[ResearchSource],
    ) -> List[str]:
        """
        Build formatted citation strings (authority-sorted callers preferred).
        Skips weak social URLs.
        """

        citations = []
        seen = set()

        for source in sources:
            url = (source.url or "").strip()
            url_l = url.lower()
            if any(
                x in url_l
                for x in (
                    "facebook.com",
                    "instagram.com",
                    "tiktok.com",
                    "reddit.com",
                )
            ):
                continue

            title = (source.title or "").strip()
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
