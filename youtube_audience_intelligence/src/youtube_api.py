"""Small, beginner-friendly wrapper around the official YouTube Data API v3."""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

import requests

API_BASE = "https://www.googleapis.com/youtube/v3"


class YouTubeAPIError(Exception):
    """A readable error returned by the YouTube Data API or URL validation."""


def parse_youtube_url(url: str) -> dict:
    """Return {'type': 'video'|'channel', 'id': value} for supported YouTube URLs."""
    value = (url or "").strip()
    if not value:
        raise YouTubeAPIError("Please paste a YouTube video or channel URL.")

    # Accept a bare channel ID as a convenience, but not arbitrary text.
    if re.fullmatch(r"UC[A-Za-z0-9_-]{20,}" , value):
        return {"type": "channel_id", "id": value}

    candidate = value if "://" in value else "https://" + value
    parsed = urlparse(candidate)
    host = (parsed.netloc or "").lower().split(":")[0]
    allowed_hosts = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "www.youtu.be"}
    if host not in allowed_hosts:
        raise YouTubeAPIError("This does not look like a YouTube URL. Use youtube.com or youtu.be.")

    path_parts = [part for part in parsed.path.split("/") if part]
    query = parse_qs(parsed.query)

    if host.endswith("youtu.be"):
        if path_parts:
            video_id = path_parts[0]
            if re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
                return {"type": "video", "id": video_id}
        raise YouTubeAPIError("The youtu.be link does not contain a valid video ID.")

    if parsed.path.rstrip("/") == "/watch":
        video_id = query.get("v", [""])[0]
        if re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
            return {"type": "video", "id": video_id}
        raise YouTubeAPIError("The video URL is missing a valid ?v= video ID.")

    if path_parts and path_parts[0] in {"shorts", "embed", "live"}:
        if len(path_parts) > 1 and re.fullmatch(r"[A-Za-z0-9_-]{11}", path_parts[1]):
            return {"type": "video", "id": path_parts[1]}
        raise YouTubeAPIError("The YouTube video link does not contain a valid video ID.")

    if path_parts and path_parts[0] == "channel" and len(path_parts) > 1:
        return {"type": "channel_id", "id": path_parts[1]}

    if path_parts and path_parts[0].startswith("@"):
        return {"type": "channel_handle", "id": path_parts[0]}

    if path_parts and path_parts[0] == "user" and len(path_parts) > 1:
        return {"type": "channel_username", "id": path_parts[1]}

    if path_parts and path_parts[0] == "c" and len(path_parts) > 1:
        return {"type": "channel_custom", "id": path_parts[1]}

    raise YouTubeAPIError(
        "Unsupported YouTube link. Paste a video URL, /@handle channel URL, or /channel/UC... URL."
    )


class YouTubeClient:
    def __init__(self, api_key: str, timeout: int = 25):
        self.api_key = (api_key or "").strip()
        self.timeout = timeout
        if not self.api_key:
            raise YouTubeAPIError("Add a YouTube Data API key in the sidebar or use CSV upload.")

    def _get(self, endpoint: str, params: dict) -> dict:
        request_params = dict(params)
        request_params["key"] = self.api_key
        try:
            response = requests.get(
                f"{API_BASE}/{endpoint}", params=request_params, timeout=self.timeout
            )
        except requests.RequestException as exc:
            raise YouTubeAPIError(f"Could not connect to YouTube: {exc}") from exc

        try:
            payload = response.json()
        except ValueError:
            payload = {}

        if not response.ok:
            error = payload.get("error", {}) if isinstance(payload, dict) else {}
            details = error.get("errors", [])
            reason = details[0].get("reason", "") if details else ""
            message = error.get("message", response.text[:250] or "Unknown API error")
            friendly = {
                "keyInvalid": "The API key is invalid. Check the key and try again.",
                "quotaExceeded": "YouTube API quota exceeded. Try again later or upload CSV files.",
                "commentsDisabled": "Comments are disabled for this video.",
                "videoNotFound": "The video was not found or is not publicly accessible.",
                "channelNotFound": "The channel was not found. Check the channel URL.",
                "forbidden": "YouTube did not allow this request. Check API access and the video's public availability.",
            }
            raise YouTubeAPIError(friendly.get(reason, f"YouTube API error ({response.status_code}): {message}"))
        return payload

    def _get_channel(self, parsed: dict) -> dict:
        kind = parsed["type"]
        key = {"channel_id": "id", "channel_handle": "forHandle", "channel_username": "forUsername"}.get(kind)
        if kind == "channel_custom":
            # Legacy /c/ URLs do not have a direct channels.list filter. Resolve the
            # custom name through search.list, then retrieve the full channel record.
            # search.list has a higher quota cost than channels.list.
            search = self._get("search", {"part": "snippet", "type": "channel", "q": parsed["id"], "maxResults": 1})
            search_items = search.get("items", [])
            channel_id = search_items[0].get("snippet", {}).get("channelId") if search_items else None
            if not channel_id:
                raise YouTubeAPIError("No channel matched that custom URL. Try the channel's /@handle link.")
            payload = self._get("channels", {"part": "snippet,contentDetails", "id": channel_id, "maxResults": 1})
        elif key is not None:
            payload = self._get("channels", {"part": "snippet,contentDetails", key: parsed["id"], "maxResults": 1})
        else:
            raise YouTubeAPIError("Unsupported channel URL. Use /@handle, /channel/UC..., /user/... or /c/...")
        items = payload.get("items", [])
        if not items:
            raise YouTubeAPIError("No public channel matched that URL. Check the link and try again.")
        return items[0]

    def _get_channel_video_ids(self, uploads_playlist_id: str, max_videos: int) -> list[str]:
        video_ids: list[str] = []
        page_token = None
        while len(video_ids) < max_videos:
            remaining = max_videos - len(video_ids)
            params = {
                "part": "contentDetails",
                "playlistId": uploads_playlist_id,
                "maxResults": min(50, remaining),
            }
            if page_token:
                params["pageToken"] = page_token
            payload = self._get("playlistItems", params)
            for item in payload.get("items", []):
                video_id = item.get("contentDetails", {}).get("videoId")
                if video_id:
                    video_ids.append(video_id)
            page_token = payload.get("nextPageToken")
            if not page_token or not payload.get("items"):
                break
        return video_ids[:max_videos]

    def _get_video_details(self, video_ids: list[str]) -> list[dict]:
        rows = []
        # videos.list accepts up to 50 IDs per request.
        for start in range(0, len(video_ids), 50):
            batch = video_ids[start : start + 50]
            payload = self._get(
                "videos",
                {"part": "snippet,statistics,contentDetails", "id": ",".join(batch)},
            )
            for item in payload.get("items", []):
                snippet = item.get("snippet", {})
                stats = item.get("statistics", {})
                details = item.get("contentDetails", {})
                thumbnails = snippet.get("thumbnails", {})
                thumbnail = (
                    thumbnails.get("maxres") or thumbnails.get("high") or
                    thumbnails.get("medium") or thumbnails.get("default") or {}
                )
                rows.append(
                    {
                        "video_id": item.get("id", ""),
                        "title": snippet.get("title", "Untitled video"),
                        "channel_title": snippet.get("channelTitle", ""),
                        "published_at": snippet.get("publishedAt", ""),
                        "thumbnail_url": thumbnail.get("url", ""),
                        "video_url": f"https://www.youtube.com/watch?v={item.get('id', '')}",
                        "views": int(stats.get("viewCount", 0) or 0),
                        "likes": int(stats["likeCount"]) if "likeCount" in stats else None,
                        "comments_count": int(stats.get("commentCount", 0) or 0),
                        "duration": details.get("duration", ""),
                        "description": snippet.get("description", ""),
                        "data_source": "YouTube Data API",
                    }
                )
        return rows

    def _get_comments(self, video_id: str, max_comments: int) -> list[dict]:
        comments = []
        page_token = None
        while len(comments) < max_comments:
            remaining = max_comments - len(comments)
            params = {
                "part": "snippet",
                "videoId": video_id,
                "maxResults": min(100, remaining),
                "order": "relevance",
                "textFormat": "plainText",
            }
            if page_token:
                params["pageToken"] = page_token
            try:
                payload = self._get("commentThreads", params)
            except YouTubeAPIError as exc:
                # Comments are optional; keep video statistics when they are disabled.
                message = str(exc).lower()
                if "comments are disabled" in message or "not allow" in message or "not found" in message:
                    return comments
                raise

            for item in payload.get("items", []):
                top = item.get("snippet", {}).get("topLevelComment", {})
                snippet = top.get("snippet", {})
                comments.append(
                    {
                        "comment_id": top.get("id", item.get("id", "")),
                        "video_id": video_id,
                        # Do not collect public display names; they are not needed for behaviour clustering.
                        "author": "",
                        "comment_text": snippet.get("textDisplay", ""),
                        "published_at": snippet.get("publishedAt", ""),
                        "like_count": int(snippet.get("likeCount", 0) or 0),
                        "reply_count": int(item.get("snippet", {}).get("totalReplyCount", 0) or 0),
                        "data_source": "YouTube Data API",
                    }
                )
                if len(comments) >= max_comments:
                    break
            page_token = payload.get("nextPageToken")
            if not page_token or not payload.get("items"):
                break
        return comments[:max_comments]

    def analyze_url(self, url: str, max_videos: int = 20, comments_per_video: int = 50) -> tuple[list[dict], list[dict], str]:
        """Download public video metadata and a capped number of public top-level comments."""
        parsed = parse_youtube_url(url)
        if parsed["type"] == "video":
            video_ids = [parsed["id"]]
            source_name = "YouTube video"
        else:
            channel = self._get_channel(parsed)
            source_name = channel.get("snippet", {}).get("title", "YouTube channel")
            uploads = channel.get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads")
            if not uploads:
                raise YouTubeAPIError("YouTube did not return an uploads playlist for this channel.")
            video_ids = self._get_channel_video_ids(uploads, int(max_videos))
            if not video_ids:
                raise YouTubeAPIError("No public videos were found for this channel.")

        videos = self._get_video_details(video_ids)
        comments = []
        for video in videos:
            if comments_per_video > 0:
                comments.extend(self._get_comments(video["video_id"], int(comments_per_video)))
        return videos, comments, source_name
