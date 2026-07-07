"""
Reddit Search Tool

Responsibilities
----------------
- Search Reddit posts
- Extract post content
- Extract top comments
- Detect images/videos
- Extract external links
- Return structured data
"""

from typing import List, Dict
import requests
import re


class RedditSearch:

    SEARCH_URL = "https://www.reddit.com/search.json"

    HEADERS = {
        "User-Agent": "EditorialAI/1.0"
    }

    def search(
        self,
        query: str,
        limit: int = 5,
        top_comments: int = 3
    ) -> List[Dict]:

        params = {
            "q": query,
            "limit": limit,
            "sort": "relevance"
        }

        try:

            response = requests.get(
                self.SEARCH_URL,
                headers=self.HEADERS,
                params=params,
                timeout=20
            )

            response.raise_for_status()

            data = response.json()

            results = []

            for child in data["data"]["children"]:

                post = child["data"]

                permalink = "https://reddit.com" + post["permalink"]

                comments = self._fetch_comments(
                    permalink + ".json",
                    top_comments
                )

                results.append({

                    "title": post.get("title"),

                    "content": post.get("selftext", ""),

                    "subreddit": post.get("subreddit"),

                    "author": post.get("author"),

                    "score": post.get("score"),

                    "comments_count": post.get("num_comments"),

                    "url": permalink,

                    "image_urls": self._extract_images(post),

                    "video_url": self._extract_video(post),

                    "external_links": self._extract_links(
                        post.get("selftext", "")
                    ),

                    "top_comments": comments

                })

            return results

        except Exception as e:

            print(f"[RedditSearch] {e}")

            return []

    def _fetch_comments(
        self,
        url: str,
        limit: int
    ) -> List[str]:

        try:

            response = requests.get(
                url,
                headers=self.HEADERS,
                timeout=20
            )

            response.raise_for_status()

            data = response.json()

            comments = []

            if len(data) > 1:

                children = data[1]["data"]["children"]

                for child in children:

                    if child["kind"] != "t1":
                        continue

                    body = child["data"].get("body")

                    if body:
                        comments.append(body)

                    if len(comments) >= limit:
                        break

            return comments

        except Exception:

            return []

    def _extract_images(
        self,
        post: Dict
    ) -> List[str]:

        images = []

        url = post.get("url_overridden_by_dest")

        if url:

            if url.endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".gif",
                    ".webp"
                )
            ):
                images.append(url)

        return images

    def _extract_video(
        self,
        post: Dict
    ):

        media = post.get("media")

        if not media:
            return None

        reddit_video = media.get("reddit_video")

        if reddit_video:
            return reddit_video.get("fallback_url")

        return None

    def _extract_links(
        self,
        text: str
    ) -> List[str]:

        pattern = r"https?://[^\s]+"

        return re.findall(pattern, text)