"""CSV readers, schema cleanup and demo-data generation."""

from __future__ import annotations

from io import BytesIO, StringIO
import re

import numpy as np
import pandas as pd

VIDEO_COLUMNS = [
    "video_id", "title", "channel_title", "published_at", "thumbnail_url", "video_url",
    "views", "likes", "comments_count", "duration", "description", "data_source",
]
COMMENT_COLUMNS = [
    "comment_id", "video_id", "author", "comment_text", "published_at", "like_count", "reply_count", "data_source",
]


def clean_column_names(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [re.sub(r"[^a-z0-9]+", "_", str(col).strip().lower()).strip("_") for col in result.columns]
    aliases = {
        "id": "video_id", "video": "video_id", "video_title": "title", "view_count": "views",
        "like_count_video": "likes", "likes_count": "likes", "comment_count": "comments_count",
        "comment": "comment_text", "text": "comment_text", "author_name": "author",
        "published": "published_at", "thumbnail": "thumbnail_url", "url": "video_url",
    }
    return result.rename(columns={key: value for key, value in aliases.items() if key in result.columns})


def _ensure_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for col in columns:
        if col not in result.columns:
            result[col] = None
    return result[columns]


def normalize_videos(frame: pd.DataFrame) -> pd.DataFrame:
    result = clean_column_names(frame)
    if "video_id" not in result.columns:
        if "video_url" in result.columns:
            result["video_id"] = result["video_url"].astype(str).str.extract(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})", expand=False)
        else:
            raise ValueError("Videos CSV needs a video_id column (or a video_url column containing YouTube links).")
    if "title" not in result.columns:
        result["title"] = result["video_id"].astype(str)
    for col in ["views", "likes", "comments_count"]:
        if col not in result.columns:
            result[col] = 0
        result[col] = pd.to_numeric(result[col], errors="coerce")
    result["views"] = result["views"].fillna(0).clip(lower=0)
    result["comments_count"] = result["comments_count"].fillna(0).clip(lower=0)
    # Keep missing likes as missing; some public like counts may be unavailable.
    result["likes"] = result["likes"].clip(lower=0)
    for col in ["channel_title", "published_at", "thumbnail_url", "video_url", "duration", "description", "data_source"]:
        if col not in result.columns:
            result[col] = ""
    result["video_id"] = result["video_id"].fillna("").astype(str)
    result = result[result["video_id"].str.len() > 0].drop_duplicates("video_id", keep="first")
    if result.empty:
        raise ValueError("No valid video rows were found in this CSV.")
    result["views"] = result["views"].astype(int)
    result["comments_count"] = result["comments_count"].astype(int)
    return _ensure_columns(result, VIDEO_COLUMNS).reset_index(drop=True)


def normalize_comments(frame: pd.DataFrame) -> pd.DataFrame:
    result = clean_column_names(frame)
    if "comment_text" not in result.columns:
        raise ValueError("Comments CSV needs a comment_text column.")
    if "video_id" not in result.columns:
        raise ValueError("Comments CSV needs a video_id column to connect comments to videos.")
    for col in ["comment_id", "author", "published_at", "data_source"]:
        if col not in result.columns:
            result[col] = ""
    for col in ["like_count", "reply_count"]:
        if col not in result.columns:
            result[col] = 0
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0).clip(lower=0).astype(int)
    result["comment_text"] = result["comment_text"].fillna("").astype(str).str.strip()
    result["video_id"] = result["video_id"].fillna("").astype(str)
    result = result[(result["comment_text"] != "") & (result["video_id"] != "")]
    return _ensure_columns(result, COMMENT_COLUMNS).reset_index(drop=True)


def load_csv_uploads(videos_file=None, comments_file=None, combined_file=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load either separate video/comment CSVs or one combined comments-and-videos CSV."""
    videos = pd.DataFrame(columns=VIDEO_COLUMNS)
    comments = pd.DataFrame(columns=COMMENT_COLUMNS)

    if combined_file is not None:
        combined = pd.read_csv(combined_file)
        cleaned = clean_column_names(combined)
        if "comment_text" not in cleaned.columns:
            videos = normalize_videos(cleaned)
        else:
            videos = normalize_videos(cleaned.drop(columns=[c for c in ["comment_text", "comment_id", "author", "published_at_comment", "like_count", "reply_count"] if c in cleaned.columns]))
            comment_frame = cleaned.copy()
            if "published_at_comment" in comment_frame.columns and "published_at" not in comment_frame.columns:
                comment_frame = comment_frame.rename(columns={"published_at_comment": "published_at"})
            comments = normalize_comments(comment_frame)
    else:
        if videos_file is not None:
            videos = normalize_videos(pd.read_csv(videos_file))
        if comments_file is not None:
            comments = normalize_comments(pd.read_csv(comments_file))

    if videos.empty:
        raise ValueError("Upload a videos CSV, or a combined CSV containing video and comment columns.")
    if not comments.empty:
        valid_ids = set(videos["video_id"].astype(str))
        comments = comments[comments["video_id"].astype(str).isin(valid_ids)].reset_index(drop=True)
    return videos, comments


def _synthetic_demo_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create visibly-labelled synthetic records for trying the app without an API key."""
    rng = np.random.default_rng(2026)
    video_topics = [
        ("New Collection Preview | Styling Ideas", "Product reveal", 42000, 3100, 280),
        ("Lawn Fabric Close-up and Quality Review", "Product quality", 26000, 1850, 390),
        ("How to Style 3 Everyday Outfits", "How-to", 56000, 4200, 210),
        ("Big Sale Announcement - Limited Offer", "Promotion", 88000, 5700, 940),
        ("Customer Unboxing and Review", "Customer story", 19000, 1450, 180),
        ("New Arrivals This Week", "Product reveal", 35000, 2100, 250),
        ("Size Guide: Find Your Perfect Fit", "Educational", 23000, 1750, 510),
        ("Behind the Scenes: Making the Collection", "Behind the scenes", 12000, 720, 65),
        ("Festive Lookbook | 5 Outfit Ideas", "Lookbook", 72000, 5200, 410),
        ("Shipping and Order Update", "Customer update", 14500, 620, 360),
        ("Budget Picks Under PKR 5,000", "Promotion", 61000, 4400, 780),
        ("New Colour Options - Which One Wins?", "Engagement prompt", 28000, 2500, 620),
    ]
    videos = []
    comments = []
    templates = {
        "price": ["Price please?", "What is the price in PKR?", "Is ki price kitni hai?", "Please share the rate."],
        "discount": ["Any discount on this?", "Will there be a sale offer?", "Thora discount kar dein please.", "Do you have a promo code?"],
        "delivery": ["Delivery kitne din mein hogi?", "Do you deliver to Karachi?", "My parcel is late, please update.", "Is shipping available nationwide?"],
        "availability": ["Is this available in medium?", "Available in Lahore?", "Is the blue colour in stock?", "Where can I find this?"],
        "purchase": ["I want to order this.", "Please send the link in inbox.", "How can I buy this?", "Interested, share details please."],
        "quality": ["How is the fabric quality?", "Please show a close-up of the material.", "Is the lawn fabric soft?", "What sizes are available?"],
        "complaint": ["My order is still waiting.", "The parcel arrived damaged.", "Delivery is too late, please help.", "I am disappointed with the service."],
        "positive": ["Very beautiful collection!", "Love this design, amazing.", "Bohat pyara suit hai.", "Quality looks great, mashallah!"],
    }
    topic_mix = ["price", "discount", "delivery", "availability", "purchase", "quality", "complaint", "positive"]
    topic_weights = np.array([.16, .12, .14, .12, .15, .12, .08, .11])
    for i, (title, category, views, likes, comment_count) in enumerate(video_topics, start=1):
        video_id = f"DEMO{i:07d}"
        # Use synthetic IDs rather than real YouTube IDs; generated links are intentionally blank.
        videos.append({
            "video_id": video_id, "title": title, "channel_title": "Demo Fashion Channel",
            "published_at": f"2026-10-{max(1, 30 - i):02d}T10:00:00Z", "thumbnail_url": "",
            "video_url": "", "views": views, "likes": likes, "comments_count": comment_count,
            "duration": "PT1M30S", "description": category, "data_source": "SIMULATED DEMO DATA",
        })
        n_comments = max(25, int(comment_count * .16))
        for j in range(n_comments):
            topic = rng.choice(topic_mix, p=topic_weights)
            text = rng.choice(templates[topic])
            # Slight content-specific lift makes the clusters demonstrable.
            if category == "Promotion" and topic == "discount":
                text = "Any special discount or sale offer on this?"
            comments.append({
                "comment_id": f"D{i:02d}C{j:04d}", "video_id": video_id,
                "author": f"Demo viewer {j + 1}", "comment_text": text,
                "published_at": f"2026-11-{(j % 15) + 1:02d}T12:00:00Z",
                "like_count": int(rng.integers(0, 12)), "reply_count": int(rng.integers(0, 4)),
                "data_source": "SIMULATED DEMO DATA",
            })
    return normalize_videos(pd.DataFrame(videos)), normalize_comments(pd.DataFrame(comments))


def get_demo_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return _synthetic_demo_data()


def dataframe_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")
