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

        Multi-query fan-out + deep stats fetch so any user topic gets
        more relevant statistics (not just a single thin snippet search).
        """
        namespace = brand_context.get("namespace", "")

        logger.info(
            "ResearchService.run() | query='%s' | namespace='%s'",
            query,
            namespace,
        )

        search_queries = self._build_search_queries(query, brand_context)
        primary_query = search_queries[0] if search_queries else (query or "").strip()

        # Retrieve internal knowledge (KB wiring optional / currently empty).
        kb_docs: List[ResearchDocument] = []
        kb_sources: List[ResearchSource] = []

        logger.info(
            "KB search complete | docs=%d | sources=%d",
            len(kb_docs),
            len(kb_sources),
        )

        # Primary: full web stack (Reddit skipped — too noisy for stats).
        web_docs, web_sources = self._search_web(
            primary_query,
            include_reddit=False,
        )

        # Secondary queries: lighter Tavily + News only (latency-safe).
        for sq in search_queries[1:]:
            extra_docs, extra_sources = self._search_web_light(sq)
            web_docs.extend(extra_docs)
            web_sources.extend(extra_sources)

        # Deep-fetch for statistics: advanced Tavily with page text.
        deep_docs, deep_sources = self._search_stats_deep(
            search_queries[1] if len(search_queries) > 1 else primary_query
        )
        web_docs.extend(deep_docs)
        web_sources.extend(deep_sources)

        # News-incident path: case briefs need reported events, not only NCRB totals.
        incident_docs = []  # type: List[ResearchDocument]
        incident_sources = []  # type: List[ResearchSource]
        if self._wants_news_incidents(query):
            incident_queries = self._build_incident_queries(query)
            for iq in incident_queries:
                idocs, isrcs = self._search_news_incidents(iq)
                incident_docs.extend(idocs)
                incident_sources.extend(isrcs)
            web_docs.extend(incident_docs)
            web_sources.extend(incident_sources)
            logger.info(
                "News-incident search | queries=%d | docs=%d",
                len(incident_queries),
                len(incident_docs),
            )

        logger.info(
            "Web search complete | docs=%d | sources=%d | queries=%d",
            len(web_docs),
            len(web_sources),
            len(search_queries),
        )

        all_docs = self._dedupe_documents(kb_docs + web_docs)
        all_sources = self._dedupe_sources(kb_sources + web_sources)

        # Rank: authority + overlap with user topic
        topic_tokens = self._topic_tokens(query)
        all_docs = sorted(
            all_docs,
            key=lambda d: (
                float((d.metadata or {}).get("authority") or 0.0)
                + 0.15 * self._text_topic_overlap(
                    f"{d.title or ''} {d.text or ''}",
                    topic_tokens,
                )
            ),
            reverse=True,
        )

        max_docs = max(5, int(getattr(settings, "MAX_RESEARCH_RESULTS", 10) or 10))
        # Keep a slightly larger pool for stats extraction, then cap stored docs
        stats_pool = all_docs[: max(max_docs * 2, 16)]
        all_docs = all_docs[:max_docs]

        logger.info(
            "Research summary | KB docs=%d | Web docs(after dedupe/cap)=%d | queries=%s",
            len(kb_docs),
            len(all_docs),
            search_queries,
        )

        unique_sources = sorted(
            all_sources,
            key=lambda s: self._get_source_authority(s.url or ""),
            reverse=True,
        )

        statistics = self._extract_statistics(stats_pool, query=query)
        citations = self._extract_citations(unique_sources)
        # Prefer the dedicated incident pool; fall back to all docs tagged as news.
        incident_pool = incident_docs or [
            d
            for d in stats_pool
            if (d.source_type or "").lower() == "news"
            or "news" in str((d.metadata or {}).get("provider") or "").lower()
        ]
        incidents = self._extract_news_incidents(incident_pool, query=query)
        # Surface top incidents inside statistics so older prompt paths still see them.
        for inc in incidents[:6]:
            tag = f"NEWS CASE: {inc}"
            if tag not in statistics:
                statistics.insert(0, tag)

        research_data = ResearchData(
            documents=all_docs,
            total_documents=len(all_docs),
            sources=unique_sources,
            statistics=statistics[:16],
            citations=citations,
            incidents=incidents,
        )

        logger.info(
            "Research complete | query='%s' | documents=%d | sources=%d | stats=%d | incidents=%d",
            query,
            len(all_docs),
            len(unique_sources),
            len(statistics),
            len(incidents),
        )

        return research_data.to_state_dict()

    # ------------------------------------------------------------------
    # Query planning / dedupe helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _topic_tokens(text: str) -> List[str]:
        stop = {
            "write", "article", "about", "the", "and", "for", "with", "from",
            "that", "this", "have", "been", "where", "please", "content",
            "blog", "post", "make", "create", "generate", "want", "need",
            "like", "very", "good", "site", "reddit", "facebook", "https",
            "http", "www", "html", "statistics", "report", "survey", "com",
            "org", "net", "findings", "research", "data",
        }
        # Drop search operators before tokenizing
        cleaned = re.sub(r"-site:\S+", " ", (text or "").lower())
        cleaned = re.sub(r"\b(?:or|and)\b", " ", cleaned)
        tokens = [
            w
            for w in re.findall(r"[a-z0-9]{3,}", cleaned)
            if w not in stop
        ]
        # Deduplicate preserving order
        out: List[str] = []
        seen = set()
        for t in tokens:
            if t not in seen:
                seen.add(t)
                out.append(t)
            if len(out) >= 16:
                break
        return out

    @classmethod
    def _text_topic_overlap(cls, text: str, tokens: List[str]) -> float:
        if not tokens:
            return 0.0
        hay = (text or "").lower()
        hits = sum(1 for t in tokens if t in hay)
        return hits / max(1, len(tokens))

    @staticmethod
    def _wants_news_incidents(query: str) -> bool:
        """True when the brief asks for cases / reported incidents, not only aggregates."""
        q = (query or "").lower()
        return bool(
            re.search(
                r"\b("
                r"cases?|incidents?|abus\w*|assault|booked|arrested|fir|"
                r"creche|crèche|daycare|day\s*care|nanny|babysitter|caregiver|"
                r"state[- ]wise|reported|between\s+20\d{2}|from\s+20\d{2}"
                r")\b",
                q,
            )
        )

    def _build_incident_queries(self, query: str) -> List[str]:
        """
        News-style queries that match what users find on Google News
        (specific incidents), not only NCRB aggregate reports.
        """
        core = (query or "").split("|")[0].strip()
        core = re.sub(r"\s*-site:\S+", " ", core, flags=re.I)
        core = re.sub(r"\s{2,}", " ", core).strip()[:220]
        years = re.findall(r"\b(20[12]\d)\b", core)
        if len(years) >= 2:
            y0, y1 = min(years), max(years)
            year_span = f"{y0} OR {y1}"
            # Also include mid years lightly for RSS
            year_span = " OR ".join(
                str(y) for y in range(int(y0), min(int(y1), int(y0) + 6) + 1)
            )
        elif years:
            year_span = " OR ".join(sorted(set(years)))
        else:
            year_span = "2020 OR 2021 OR 2022 OR 2023 OR 2024 OR 2025 OR 2026"

        india = "India" if re.search(r"\b(india|indian)\b", core, re.I) else ""
        # Topic cores for childcare abuse briefs (nanny/nannies/ayah/creche…)
        if re.search(
            r"\b(nann(?:y|ies)|babysitters?|caregivers?|ayahs?|"
            r"creche|crèche|day\s*care|daycare)\b",
            core,
            re.I,
        ):
            bases = [
                f"nanny abuse children {india} {year_span}".strip(),
                f"nannies booked child abuse {india} toddlers OR kids".strip(),
                f"creche OR daycare abuse nanny {india} Bengaluru OR Mumbai OR Delhi OR Hyderabad".strip(),
                f"domestic help OR maid OR ayah child abuse case {india} police".strip(),
                f"Capgemini creche OR Bengaluru daycare toddler abuse {india}".strip(),
            ]
        else:
            # Keep incident queries short — long blog prompts hurt News RSS recall
            short = re.sub(
                r"\b(write|an|article|about|the|should|have|add|angle|how|can|"
                r"help|avoid|such|incidents?|between)\b",
                " ",
                core,
                flags=re.I,
            )
            short = re.sub(r"\s{2,}", " ", short).strip()[:140] or core[:140]
            bases = [
                f"{short} news cases {year_span}".strip(),
                f"{short} police booked OR arrested {india}".strip(),
            ]

        out: List[str] = []
        seen = set()
        for q in bases:
            q = re.sub(r"\s{2,}", " ", q).strip()[:500]
            key = q.lower()
            if q and key not in seen:
                seen.add(key)
                out.append(q)
            if len(out) >= 4:
                break
        return out

    def _search_news_incidents(
        self,
        query: str,
    ) -> Tuple[List[ResearchDocument], List[ResearchSource]]:
        """
        Prioritize Google News + news-domain Tavily for reported incidents.
        """
        documents: List[ResearchDocument] = []
        sources: List[ResearchSource] = []
        if not (query or "").strip():
            return documents, sources

        # 1) Google News RSS — closest to what users see in Google News
        try:
            gnews = GoogleNewsRSS()
            news_results = gnews.search(query, max_results=10)
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
                body = text_content
                if title and title not in body:
                    body = f"{title}. {body}"
                url = item.get("url") or ""
                documents.append(
                    ResearchDocument(
                        text=body,
                        title=title,
                        url=url,
                        source_type="news",
                        relevance_score=0.92,
                        metadata={
                            "source_name": item.get("source") or "",
                            "provider": "google_news_incidents",
                            "published_at": item.get("published_at") or "",
                            "authority": max(
                                0.85,
                                self._get_source_authority(url),
                            ),
                            "is_incident": True,
                        },
                    )
                )
                sources.append(
                    ResearchSource(
                        title=title,
                        url=url,
                        source_type="news",
                        published_date=item.get("published_at"),
                        author=item.get("source") or "",
                        snippet=(item.get("description") or "")[:300],
                    )
                )
        except Exception as exc:
            logger.error("Incident Google News failed: %s", exc, exc_info=True)

        # 2) Tavily biased to major Indian news domains
        if settings.TAVILY_API_KEY:
            try:
                tavily = TavilySearch()
                tq = (
                    f"{query} "
                    f"(site:hindustantimes.com OR site:timesofindia.indiatimes.com "
                    f"OR site:indianexpress.com OR site:thehindu.com "
                    f"OR site:livemint.com OR site:ndtv.com)"
                )
                results = tavily.search(
                    tq,
                    max_results=min(5, settings.TAVILY_MAX_RESULTS),
                    search_depth="advanced",
                    include_raw_content=True,
                    include_answer=False,
                )
                for item in results:
                    text_content = (
                        item.get("raw_content")
                        or item.get("content")
                        or ""
                    ).strip()
                    if not text_content:
                        continue
                    url = item.get("url") or ""
                    # Skip social even if Tavily returns it
                    if any(
                        x in url.lower()
                        for x in ("facebook.com", "instagram.com", "reddit.com")
                    ):
                        continue
                    documents.append(
                        ResearchDocument(
                            text=text_content[:6000],
                            title=item.get("title") or "",
                            url=url,
                            source_type="news",
                            relevance_score=float(item.get("score") or 0.8),
                            metadata={
                                "provider": "tavily_news_incidents",
                                "authority": max(
                                    0.85,
                                    self._get_source_authority(url),
                                ),
                                "is_incident": True,
                            },
                        )
                    )
                    sources.append(
                        ResearchSource(
                            title=item.get("title") or "",
                            url=url,
                            source_type="news",
                            snippet=(item.get("content") or "")[:300],
                        )
                    )
            except Exception as exc:
                logger.error("Incident Tavily news failed: %s", exc, exc_info=True)

        return documents, sources

    def _extract_news_incidents(
        self,
        documents: List[ResearchDocument],
        query: str = "",
    ) -> List[str]:
        """
        Turn news docs into concise incident lines for the Writer.
        Prefers titles that look like reported cases over generic opinion.
        """
        topic_tokens = self._topic_tokens(query)
        scored: List[tuple] = []
        seen = set()

        case_signals = re.compile(
            r"\b("
            r"abus\w*|assault|booked|arrested|fir|police|accused|charged|"
            r"creche|crèche|daycare|nanny|caregiver|toddler|child|kids?|"
            r"horror|complaint|custody|pocso"
            r")\b",
            re.I,
        )

        for doc in documents or []:
            title = (doc.title or "").strip()
            text = (doc.text or "").strip()
            if not title and not text:
                continue
            blob = f"{title}. {text[:400]}"
            if not case_signals.search(blob):
                continue
            url = (doc.url or "").lower()
            if any(x in url for x in ("facebook.com", "instagram.com", "reddit.com")):
                continue

            auth = float((doc.metadata or {}).get("authority") or 0.7)
            overlap = self._text_topic_overlap(blob, topic_tokens)
            pub = str((doc.metadata or {}).get("published_at") or "")[:16]
            source_name = str(
                (doc.metadata or {}).get("source_name")
                or doc.title
                or "News"
            ).strip()
            # Build a writer-ready incident line
            snippet = re.sub(r"\s+", " ", text[:220]).strip()
            line = title or snippet
            if snippet and title and snippet.lower() not in title.lower():
                line = f"{title} — {snippet}"
            meta_bits = []
            if source_name and source_name.lower() not in line.lower():
                meta_bits.append(source_name[:80])
            if pub:
                meta_bits.append(pub)
            if doc.url:
                meta_bits.append(doc.url)
            if meta_bits:
                line = f"{line} ({'; '.join(meta_bits)})"

            key = title.lower() if title else line.lower()[:120]
            if key in seen:
                continue
            seen.add(key)
            score = auth + overlap + (0.2 if (doc.metadata or {}).get("is_incident") else 0)
            # Boost if year from query appears
            years = set(re.findall(r"\b(20[12]\d)\b", (query or "").lower()))
            if years and any(y in blob.lower() or y in pub for y in years):
                score += 0.25
            scored.append((score, line[:500]))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:12]]

    def _build_search_queries(
        self,
        query: str,
        brand_context: Dict,
    ) -> List[str]:
        """
        Fan out 2–3 search queries from the user topic so stats/reports
        are not missed when the raw prompt is conversational.
        """
        primary = self._enrich_query_for_market(query, brand_context)
        if not primary:
            return []

        core = primary.split("|")[0].strip()
        # Strip site operators for alternate queries (cleaner for news/stats).
        core_clean = re.sub(
            r"\s*-site:\S+",
            " ",
            core,
            flags=re.I,
        )
        core_clean = re.sub(r"\s{2,}", " ", core_clean).strip()[:280]

        queries = [primary]
        years = re.findall(r"\b(20[12]\d)\b", core_clean)
        year_bit = " OR ".join(sorted(set(years))[:4]) if years else "2023 OR 2024 OR 2025"

        # For case/incident briefs, prefer news/cases query over dry "survey/report"
        if self._wants_news_incidents(query):
            queries.append(
                f"{core_clean} news cases OR police OR booked ({year_bit})"
            )
        elif not re.search(
            r"\b(statistics?|survey|report|benchmark|market\s+size)\b",
            core_clean,
            re.I,
        ):
            queries.append(
                f"{core_clean} statistics OR survey OR report ({year_bit})"
            )
        else:
            queries.append(f"{core_clean} {year_bit}")

        # Geography-neutral data query if user named a market
        if re.search(r"\b(india|indian|usa|u\.s\.|uk|britain)\b", core_clean, re.I):
            geo_q = (
                f"{core_clean} data OR findings OR research "
                f"-site:reddit.com -site:facebook.com"
            )
            if geo_q not in queries:
                queries.append(geo_q[:500])

        # Deduplicate while preserving order
        seen = set()
        out: List[str] = []
        for q in queries:
            key = q.lower().strip()
            if key and key not in seen:
                seen.add(key)
                out.append(q[:500])
            if len(out) >= 3:
                break
        return out

    def _dedupe_documents(
        self,
        documents: List[ResearchDocument],
    ) -> List[ResearchDocument]:
        """Prefer longer / higher-authority copy of the same URL."""
        best: Dict[str, ResearchDocument] = {}
        orphan: List[ResearchDocument] = []
        for doc in documents or []:
            url = (doc.url or "").strip().lower().rstrip("/")
            if not url:
                orphan.append(doc)
                continue
            prev = best.get(url)
            if prev is None:
                best[url] = doc
                continue
            prev_auth = float((prev.metadata or {}).get("authority") or 0)
            cur_auth = float((doc.metadata or {}).get("authority") or 0)
            if len(doc.text or "") > len(prev.text or "") or cur_auth > prev_auth:
                best[url] = doc
        return list(best.values()) + orphan

    def _dedupe_sources(
        self,
        sources: List[ResearchSource],
    ) -> List[ResearchSource]:
        unique: List[ResearchSource] = []
        seen_urls = set()
        for src in sources or []:
            url = (src.url or "").strip().lower().rstrip("/")
            if url:
                if url in seen_urls:
                    continue
                seen_urls.add(url)
            unique.append(src)
        return unique

    def _search_web_light(
        self,
        query: str,
    ) -> Tuple[List[ResearchDocument], List[ResearchSource]]:
        """Secondary-query path: Tavily snippets + Google News only."""
        documents: List[ResearchDocument] = []
        sources: List[ResearchSource] = []
        if not (query or "").strip():
            return documents, sources

        try:
            tavily = TavilySearch()
            results = tavily.search(
                query,
                max_results=min(4, settings.TAVILY_MAX_RESULTS),
                search_depth="basic",
                include_raw_content=False,
                include_answer=False,
            )
            for item in results:
                text_content = (
                    item.get("raw_content")
                    or item.get("content")
                    or ""
                ).strip()
                if not text_content:
                    continue
                documents.append(
                    ResearchDocument(
                        text=text_content,
                        title=item.get("title") or "",
                        url=item.get("url") or "",
                        source_type="web",
                        relevance_score=float(item.get("score") or 0.0),
                        metadata={
                            "authority": self._get_source_authority(
                                item.get("url", "")
                            ),
                            "provider": "tavily_light",
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
            logger.error("Light Tavily search failed: %s", exc, exc_info=True)

        try:
            gnews = GoogleNewsRSS()
            news_results = gnews.search(
                query,
                max_results=min(5, getattr(settings, "GOOGLE_NEWS_MAX_RESULTS", 8)),
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
                            "provider": "google_news_rss_light",
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
            logger.error("Light Google News failed: %s", exc, exc_info=True)

        return documents, sources

    def _search_stats_deep(
        self,
        query: str,
    ) -> Tuple[List[ResearchDocument], List[ResearchSource]]:
        """
        Advanced Tavily with raw page text — best path for real statistics.
        Limited to a few results to control latency.
        """
        documents: List[ResearchDocument] = []
        sources: List[ResearchSource] = []
        if not (query or "").strip() or not settings.TAVILY_API_KEY:
            return documents, sources

        try:
            logger.info("Running deep stats Tavily search | query='%s'", query[:120])
            tavily = TavilySearch()
            results = tavily.search(
                query,
                max_results=min(4, settings.TAVILY_MAX_RESULTS),
                search_depth="advanced",
                include_raw_content=True,
                include_answer=True,
            )
            # Tavily answer (when present) often contains attributed figures
            # — surface it as a pseudo-document for regex stats extraction.
            # Note: answer may be on the raw response; wrapper may not return it.
            for item in results:
                text_content = (
                    item.get("raw_content")
                    or item.get("content")
                    or ""
                ).strip()
                if not text_content:
                    continue
                documents.append(
                    ResearchDocument(
                        text=text_content[:6000],
                        title=item.get("title") or "",
                        url=item.get("url") or "",
                        source_type="web",
                        relevance_score=float(item.get("score") or 0.75),
                        metadata={
                            "authority": max(
                                0.7,
                                self._get_source_authority(item.get("url", "")),
                            ),
                            "provider": "tavily_deep",
                        },
                    )
                )
                sources.append(
                    ResearchSource(
                        title=item.get("title") or "",
                        url=item.get("url") or "",
                        source_type="web",
                        snippet=(item.get("content") or "")[:300],
                    )
                )
            logger.info("Deep stats Tavily returned %d docs", len(documents))
        except Exception as exc:
            logger.error("Deep stats Tavily failed: %s", exc, exc_info=True)

        return documents, sources

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
                # Geo already applied above — only add noise filters here.
                search_core = (
                    f"{search_core} -site:reddit.com -site:facebook.com"
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
        include_reddit: bool = False,
    ) -> Tuple[List[ResearchDocument], List[ResearchSource]]:
        """
        Search external research sources.
        Reddit is off by default — it rarely yields reliable stats.
        """
        documents: List[ResearchDocument] = []
        sources: List[ResearchSource] = []

        logger.info(
            "Starting external research | query='%s' | reddit=%s",
            query,
            include_reddit,
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
        # 3. Reddit Search (optional — skipped by default for quality)
        # ------------------------------------------------------------------
        if include_reddit:
            try:
                logger.info(
                    "Running Reddit search via Tavily | query='%s'",
                    query,
                )

                tavily = TavilySearch()
                reddit_results = tavily.search(
                    f"site:reddit.com {query}",
                    max_results=min(3, settings.TAVILY_MAX_RESULTS),
                    search_depth="basic",
                    include_raw_content=False,
                    include_answer=False,
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
                                    "authority": 0.25,
                                },
                            )
                        )
                        sources.append(
                            ResearchSource(
                                title=item.get("title") or "",
                                url=item.get("url") or "",
                                source_type="reddit",
                                snippet=(item.get("content") or "")[:200],
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
        topic_tokens = self._topic_tokens(query)

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

            # Prefer docs that actually match the user topic
            doc_overlap = self._text_topic_overlap(
                f"{doc.title or ''} {doc.text or ''}",
                topic_tokens,
            )
            if topic_tokens and doc_overlap < 0.08 and auth < 0.9:
                # Weak topical match on non-elite sources — skip
                continue

            text = doc.text[:5000]
            title_l = (doc.title or "").lower()

            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    start = max(match.start() - 100, 0)
                    end = min(match.end() + 140, len(text))
                    snippet = text[start:end].strip()
                    # Drop ultra-short / likely false-positive numeric crumbs
                    if len(snippet) < 28:
                        continue
                    source_label = (doc.title or "").strip()
                    if source_label and source_label.lower() not in snippet.lower():
                        snippet = f"{snippet} (Source: {source_label})"
                    if not snippet or snippet in seen:
                        continue
                    seen.add(snippet)

                    snip_l = snippet.lower()
                    snip_overlap = self._text_topic_overlap(snip_l, topic_tokens)
                    score = auth + (0.5 * snip_overlap) + (0.25 * doc_overlap)
                    if years and any(y in snip_l or y in title_l for y in years):
                        score += 0.35
                    if wants_nri and re.search(r"\bnri\b|non[-\s]?resident", snip_l):
                        score += 0.4
                    elif wants_nri and re.search(
                        r"\b(adults? in india|indian adults?|three out of four)\b",
                        snip_l,
                    ):
                        score -= 0.35
                    if wants_property and re.search(
                        r"\b(property|real\s*estate|land|rental|title)\b", snip_l
                    ):
                        score += 0.25
                    # Prefer snippets that name a source org
                    if re.search(
                        r"\b(according to|report|survey|study|mckinsey|gartner|"
                        r"ncrb|world bank|imf|oecd|unicef)\b",
                        snip_l,
                    ):
                        score += 0.15
                    scored.append((score, snippet))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:12]]

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
