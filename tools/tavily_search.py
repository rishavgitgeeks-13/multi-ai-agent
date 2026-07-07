"""
Tavily Search Tool

Responsibilities
----------------
- Search the web using Tavily
- Return structured search results
- Keep raw web research separate from the LLM
"""

from typing import List, Dict

from tavily import TavilyClient

from config.settings import settings


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
        max_results: int = None
    ) -> List[Dict]:
        """
        Search the web.

        Parameters
        ----------
        query : str
            Search query.

        max_results : int
            Number of results.

        Returns
        -------
        List[Dict]
        """

        if max_results is None:
            max_results = settings.TAVILY_MAX_RESULTS

        try:

            response = self.client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
                include_answer=True,
                include_raw_content=True,
                include_images=False,
            )

            results = []

            for item in response.get("results", []):

                results.append(
                    {
                        "title": item.get("title"),

                        "content": item.get("content"),

                        "raw_content": item.get(
                            "raw_content"
                        ),

                        "url": item.get("url"),

                        "score": item.get("score"),
                    }
                )

            return results

        except Exception as e:

            print(f"[TavilySearch] {e}")

            return []