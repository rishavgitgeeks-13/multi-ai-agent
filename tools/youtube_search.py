"""
Searches YouTube for relevant videos and extracts
their transcripts for downstream AI agents.
"""

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import List, Dict

import requests
from youtube_transcript_api import YouTubeTranscriptApi

from config.settings import settings

# Hard cap so a stalled transcript fetch cannot block the whole pipeline.
_TRANSCRIPT_TIMEOUT_SEC = 8
_MAX_TRANSCRIPT_CHARS = 4000


class YouTubeSearch:

    SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"

    def __init__(self):
        self.api_key = settings.YOUTUBE_API_KEY

    def search(self, query: str, max_results: int = None) -> List[Dict]:
        """
        Search YouTube and extract transcripts.

        Returns
        -------
        [
            {
                "title": "...",
                "channel": "...",
                "description": "...",
                "published_at": "...",
                "video_id": "...",
                "url": "...",
                "transcript": "..."
            }
        ]
        """

        if max_results is None:
            max_results = settings.YOUTUBE_MAX_RESULTS

        params = {
            "part": "snippet",
            "q": query,
            "maxResults": max_results,
            "type": "video",
            "order": "relevance",
            "key": self.api_key,
        }

        try:

            response = requests.get(
                self.SEARCH_URL,
                params=params,
                timeout=20,
            )

            response.raise_for_status()

            data = response.json()

            videos = []

            for item in data.get("items", []):

                snippet = item["snippet"]

                video_id = item["id"]["videoId"]

                transcript = self._get_transcript(video_id)
                transcript = transcript[:_MAX_TRANSCRIPT_CHARS]
                print(
                    f"[YouTubeSearch] "
                    f"{video_id} "
                    f"transcript_length={len(transcript)}"
                )

                videos.append(
                    {
                        "title": snippet.get("title"),
                        "channel": snippet.get("channelTitle"),
                        "description": snippet.get("description"),
                        "published_at": snippet.get("publishedAt"),
                        "video_id": video_id,
                        "url": f"https://www.youtube.com/watch?v={video_id}",
                        "transcript": transcript,
                    }
                )

            return videos

        except Exception as e:

            print(f"[YouTubeSearch] {e}")

            return []

    def _get_transcript(self, video_id: str) -> str:
        """
        Download transcript for a video with a hard timeout.

        Returns empty string if unavailable or timed out.
        """
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._fetch_transcript, video_id)
                return future.result(timeout=_TRANSCRIPT_TIMEOUT_SEC)
        except FuturesTimeoutError:
            print(
                f"[YouTubeTranscript] "
                f"video_id={video_id} "
                f"timed out after {_TRANSCRIPT_TIMEOUT_SEC}s — skipping"
            )
            return ""
        except Exception as exc:
            print(
                f"[YouTubeTranscript] "
                f"video_id={video_id} "
                f"error={type(exc).__name__}: {exc}"
            )
            return ""

    def _fetch_transcript(self, video_id: str) -> str:
        """Blocking transcript download (run inside a timed worker)."""
        ytt_api = YouTubeTranscriptApi()

        try:
            transcript = ytt_api.fetch(
                video_id,
                languages=["en", "hi"],
            )
        except Exception:
            transcript = ytt_api.fetch(video_id)

        return " ".join(segment.text for segment in transcript)
