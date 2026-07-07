"""
News Search Tool

Responsibilities:
- Search recent news articles
- Return clean structured data
- Handle API errors gracefully
"""

from typing import List, Dict
import requests

from config.settings import settings


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

        params = {
            "q": query,
            "pageSize": page_size,
            "language": "en",
            "sortBy": "publishedAt",
            "apiKey": self.api_key,
        }

        try:
            response = requests.get(
                self.BASE_URL,
                params=params,
                timeout=15
            )

            response.raise_for_status()

            data = response.json()

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

            return articles

        except Exception as e:
            print(f"[NewsSearch] Error: {e}")
            return []