"""
Searches YouTube for relevant videos and extracts
their transcripts for downstream AI agents.
"""

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Dict, List

import requests
from youtube_transcript_api import YouTubeTranscriptApi

from config.settings import settings

# Hard cap so a stalled transcript fetch cannot block the whole pipeline.
_TRANSCRIPT_TIMEOUT_SEC = 10
_MAX_TRANSCRIPT_CHARS = 4000


class YouTubeSearch:

    SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"

    def __init__(self):
        self.api_key = settings.YOUTUBE_API_KEY

    def search(self, query: str, max_results: int = None) -> List[Dict]:
        """
        Search YouTube and extract transcripts.

        Videos without a usable transcript are still returned with
        transcript="" — ResearchService must skip those for document bodies.
        """
        if not self.api_key:
            print("[YouTubeSearch] YOUTUBE_API_KEY not configured")
            return []

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
                    f"[YouTubeSearch] {video_id} "
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
        """Download transcript with a hard timeout. Empty if unavailable."""
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._fetch_transcript, video_id)
                return future.result(timeout=_TRANSCRIPT_TIMEOUT_SEC)
        except FuturesTimeoutError:
            print(
                f"[YouTubeTranscript] video_id={video_id} "
                f"timed out after {_TRANSCRIPT_TIMEOUT_SEC}s — skipping"
            )
            return ""
        except Exception as exc:
            print(
                f"[YouTubeTranscript] video_id={video_id} "
                f"error={type(exc).__name__}: {exc}"
            )
            return ""

    def _fetch_transcript(self, video_id: str) -> str:
        """
        Blocking transcript download.
        Prefer English/Hindi; fall back to any generated track.
        """
        ytt_api = YouTubeTranscriptApi()

        # Preferred language order
        for langs in (["en", "en-US", "en-GB", "hi"], None):
            try:
                if langs:
                    transcript = ytt_api.fetch(video_id, languages=langs)
                else:
                    transcript = ytt_api.fetch(video_id)
                text = " ".join(segment.text for segment in transcript).strip()
                if text:
                    return text
            except Exception:
                continue

        # Last resort: list tracks and take the first fetchable one
        try:
            listing = ytt_api.list(video_id)
            for track in listing:
                try:
                    fetched = track.fetch()
                    text = " ".join(segment.text for segment in fetched).strip()
                    if text:
                        return text
                except Exception:
                    continue
        except Exception:
            pass

        return ""
