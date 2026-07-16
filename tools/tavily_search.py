"""
Tavily Search Tool

Responsibilities
----------------
- Search the web using Tavily
- Return structured search results
- Keep raw web research separate from the LLM
"""

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import List, Dict, Optional

from tavily import TavilyClient

from config.settings import settings

# Prevent a stalled Tavily call from blocking the whole generate request.
_TAVILY_TIMEOUT_SEC = 45
_MAX_RAW_CONTENT_CHARS = 3000


class TavilySearch:
    """
    Wrapper around the Tavily Search API.
    """

    def __init__(self):

        self.client = TavilyClient(
            api_key=settings.TAVILY_API_KEY
        )

    def search(
        self,
        query: str,
        max_results: int = None,
        search_depth: str = "basic",
        include_raw_content: bool = False,
        include_answer: bool = False,
    ) -> List[Dict]:
        """
        Search the web.

        Parameters
        ----------
        query : str
            Search query.

        max_results : int
            Number of results.

        search_depth : str
            "basic" (fast) or "advanced" (slower, deeper).

        include_raw_content : bool
            When True, Tavily scrapes full page text — much slower.

        include_answer : bool
            Whether to request Tavily's synthesized answer.

        Returns
        -------
        List[Dict]
        """

        if max_results is None:
            max_results = settings.TAVILY_MAX_RESULTS

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self._run_search,
                    query,
                    max_results,
                    search_depth,
                    include_raw_content,
                    include_answer,
                )
                return future.result(timeout=_TAVILY_TIMEOUT_SEC)

        except FuturesTimeoutError:
            print(
                f"[TavilySearch] timed out after {_TAVILY_TIMEOUT_SEC}s "
                f"| query={query[:80]!r}"
            )
            return []

        except Exception as e:

            print(f"[TavilySearch] {e}")

            return []

    def _run_search(
        self,
        query: str,
        max_results: int,
        search_depth: str,
        include_raw_content: bool,
        include_answer: bool,
    ) -> List[Dict]:

        response = self.client.search(
            query=query,
            search_depth=search_depth,
            max_results=max_results,
            include_answer=include_answer,
            include_raw_content=include_raw_content,
            include_images=False,
        )

        results = []

        for item in response.get("results", []):
            raw: Optional[str] = item.get("raw_content")
            if raw and len(raw) > _MAX_RAW_CONTENT_CHARS:
                raw = raw[:_MAX_RAW_CONTENT_CHARS]

            results.append(
                {
                    "title": item.get("title"),
                    "content": item.get("content"),
                    "raw_content": raw,
                    "url": item.get("url"),
                    "score": item.get("score"),
                }
            )

        return results
