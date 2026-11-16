from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from src.data_loader import (
    COMMENT_COLUMNS, VIDEO_COLUMNS, dataframe_csv_bytes, get_demo_data, load_csv_uploads,
)
from src.ml_models import cluster_comments, detect_video_anomalies
from src.nlp import analyze_comments
from src.recommendations import generate_recommendations
from src.youtube_api import YouTubeAPIError, YouTubeClient, parse_youtube_url


APP_DIR = Path(__file__).parent
load_dotenv(APP_DIR / ".env")

st.set_page_config(
    page_title="YouTube Audience Intelligence",
    page_icon="▶️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Clean YouTube-inspired light theme with restrained red accents.
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
      html, body, [class*="css"] { font-family: Inter, Arial, sans-serif; }
      .stApp { background: #f9f9f9; color: #171717; }
      [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e7e7e7; }
      [data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }
      .brand-row { display:flex; align-items:center; gap:10px; margin-bottom: 8px; }
      .brand-play { display:inline-flex; align-items:center; justify-content:center; background:#ff0033; color:#fff; width:34px; height:25px; border-radius:8px; font-size:14px; }
      .brand-name { font-size:17px; font-weight:800; letter-spacing:-.5px; color:#181818; }
      .app-subtitle { color:#777; font-size:11px; text-transform:uppercase; letter-spacing:1.1px; font-weight:700; margin:4px 0 18px; }
      .page-title { font-weight:800; letter-spacing:-1px; font-size:30px; margin:2px 0 3px; color:#171717; }
      .page-subtitle { color:#6b7280; margin-bottom:18px; font-size:14px; }
      .section-title { font-weight:700; font-size:18px; margin:12px 0 10px; color:#202124; }
      .metric-card { background:#fff; border:1px solid #e8e8e8; padding:16px 18px; border-radius:13px; min-height:96px; }
      .metric-label { color:#70757a; font-size:12px; font-weight:600; margin-bottom:8px; }
      .metric-value { color:#171717; font-size:26px; font-weight:800; letter-spacing:-.6px; }
      .metric-foot { color:#888; font-size:11px; margin-top:2px; }
      .video-card { background:#fff; border:1px solid #e7e7e7; border-radius:13px; padding:10px; height:100%; }
      .video-card-title { font-weight:700; font-size:13px; line-height:1.4; margin:7px 0 4px; color:#202124; }
      .video-card-meta { font-size:11px; color:#777; }
      .pill { display:inline-block; border-radius:30px; padding:4px 9px; font-size:11px; font-weight:600; background:#f1f1f1; color:#555; margin:2px 4px 2px 0; }
      .recommendation { background:#fff; border:1px solid #e8e8e8; border-left:4px solid #ff0033; border-radius:11px; padding:14px 16px; margin:8px 0; }
      .recommendation-title { font-weight:700; font-size:14px; margin-bottom:5px; }
      .recommendation-body { font-size:13px; color:#3f3f46; line-height:1.5; }
      .evidence { color:#777; font-size:11px; margin-top:7px; }
      .note { background:#fff7f7; border:1px solid #ffd7de; padding:11px 13px; border-radius:10px; color:#7a1d2b; font-size:12px; }
      .muted { color:#777; font-size:12px; }
      div[data-testid="stMetric"] { background:#fff; border:1px solid #e8e8e8; padding:13px 15px; border-radius:12px; }
      div[data-testid="stMetricLabel"] p { font-size:12px; color:#777; }
      div[data-testid="stMetricValue"] { font-size:25px; font-weight:800; }
      .stButton > button[kind="primary"] { background:#ff0033; border:1px solid #ff0033; border-radius:9px; font-weight:700; }
      .stButton > button[kind="primary"]:hover { background:#d9002b; border-color:#d9002b; }
      .stTextInput input { border-radius:9px; }
      .stTabs [data-baseweb="tab-list"] { gap:10px; }
      .stTabs [data-baseweb="tab"] { padding:8px 14px; }
      hr { border-color:#e9e9e9; }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state() -> None:
    defaults = {
        "videos": pd.DataFrame(columns=VIDEO_COLUMNS),
        "comments": pd.DataFrame(columns=COMMENT_COLUMNS),
        "analyzed_comments": pd.DataFrame(),
        "clustered_comments": pd.DataFrame(),
        "cluster_summary": pd.DataFrame(),
        "anomalies": pd.DataFrame(),
        "recommendations": [],
        "silhouette": None,
        "source_label": "No dataset loaded",
        "dataset_ready": False,
        "demo_mode": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def prepare_analysis(videos: pd.DataFrame, comments: pd.DataFrame, cluster_count: int, source_label: str, demo_mode: bool = False) -> None:
    """Run NLP, K-Means, anomaly detection and recommendations."""
    clean_videos = videos.copy()
    clean_comments = comments.copy()
    analyzed_comments = analyze_comments(clean_comments)
    clustered_comments, cluster_summary, silhouette = cluster_comments(analyzed_comments, cluster_count)
    anomalies = detect_video_anomalies(clean_videos)
    recommendations = generate_recommendations(clean_videos, clustered_comments, anomalies)

    st.session_state.videos = clean_videos
    st.session_state.comments = clean_comments
    st.session_state.analyzed_comments = analyzed_comments
    st.session_state.clustered_comments = clustered_comments
    st.session_state.cluster_summary = cluster_summary
    st.session_state.anomalies = anomalies
    st.session_state.recommendations = recommendations
    st.session_state.silhouette = silhouette
    st.session_state.source_label = source_label
    st.session_state.dataset_ready = True
    st.session_state.demo_mode = demo_mode


def metric_card(label: str, value: str, foot: str = "") -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-foot">{foot}</div></div>',
        unsafe_allow_html=True,
    )


def fmt_num(value) -> str:
    try:
        return f"{float(value):,.0f}"
    except (TypeError, ValueError):
        return "—"


def fmt_rate(value) -> str:
    try:
        return f"{float(value):.2f}%"
    except (TypeError, ValueError):
        return "—"


def show_header(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="page-title">{title}</div><div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def require_data() -> bool:
    if not st.session_state.dataset_ready:
        st.info("No analysis loaded yet. Paste a YouTube link, upload CSV files, or load the clearly-labelled demo dataset from the Home page.")
        return False
    return True


init_state()

# Sidebar navigation and API key input.
with st.sidebar:
    st.markdown(
        '<div class="brand-row"><span class="brand-play">▶</span><span class="brand-name">YouTube Intelligence</span></div>'
        '<div class="app-subtitle">Audience & Campaign Engine</div>',
        unsafe_allow_html=True,
    )
    page = st.radio(
        "NAVIGATION",
        ["Home", "Video performance", "Audience insights", "Anomalies & actions", "Data & export"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**YouTube Data API**")
    configured_api_key = os.getenv("YOUTUBE_API_KEY", "").strip()
    try:
        configured_api_key = str(st.secrets.get("YOUTUBE_API_KEY", configured_api_key)).strip()
    except Exception:
        pass
    visitor_api_key = st.text_input(
        "Your YouTube API key (optional)",
        value="",
        type="password",
        help="Only needed when the app owner has not configured a server-side key. You can also use CSV upload or demo data.",
    ).strip()
    api_key = visitor_api_key or configured_api_key
    if configured_api_key and not visitor_api_key:
        st.caption("A server-configured YouTube API key is active. It is not displayed here.")
    else:
        st.caption("Do not commit API keys to GitHub. Use Streamlit Secrets for a server-side key.")
    st.divider()
    st.caption("Current dataset")
    st.markdown(f"**{st.session_state.source_label}**")
    if st.session_state.dataset_ready:
        st.caption(f"{len(st.session_state.videos):,} videos · {len(st.session_state.comments):,} comments")
    if st.button("Clear current dataset", use_container_width=True):
        for key in ["videos", "comments", "analyzed_comments", "clustered_comments", "cluster_summary", "anomalies", "recommendations"]:
            st.session_state[key] = pd.DataFrame() if key != "recommendations" else []
        st.session_state.dataset_ready = False
        st.session_state.source_label = "No dataset loaded"
        st.session_state.demo_mode = False
        st.rerun()


if page == "Home":
    show_header("Understand your YouTube audience", "Turn video performance and public comments into practical campaign decisions.")
    st.markdown(
        '<div class="note">YouTube only · Uses public video metadata and public top-level comments. '
        'Audience segments describe comment behaviour, not verified people, demographics or unique viewers.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("### Start with a YouTube channel or video")
    with st.form("youtube_analysis_form"):
        youtube_url = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/@channelname  or  https://www.youtube.com/watch?v=...",
            help="Supported: video, Shorts, /@handle, /channel/UC... and /user/ links.",
        )
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            max_videos = st.slider("Videos from a channel", min_value=1, max_value=50, value=15)
        with c2:
            comments_per_video = st.slider("Top-level comments per video", min_value=0, max_value=100, value=40, step=10)
        with c3:
            cluster_count = st.slider("Audience clusters (K)", min_value=2, max_value=8, value=5)
        submitted = st.form_submit_button("Analyze YouTube content", type="primary", use_container_width=True)

    if submitted:
        try:
            parsed = parse_youtube_url(youtube_url)
            if not api_key.strip():
                st.error("Add a YouTube Data API key in the left sidebar, or use CSV upload / demo data below.")
            else:
                with st.spinner("Fetching public YouTube data and building the ML analysis..."):
                    client = YouTubeClient(api_key)
                    video_rows, comment_rows, source_name = client.analyze_url(
                        youtube_url, max_videos=max_videos, comments_per_video=comments_per_video
                    )
                    prepare_analysis(
                        pd.DataFrame(video_rows), pd.DataFrame(comment_rows), cluster_count,
                        f"YouTube API · {source_name}", demo_mode=False,
                    )
                st.success(f"Analysis ready: {len(video_rows)} video(s) and {len(comment_rows)} public top-level comment(s).")
                st.rerun()
        except YouTubeAPIError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"The analysis could not be completed: {exc}")

    st.markdown("### No API key? Try the alternatives")
    left, right = st.columns([1, 1])
    with left:
        if st.button("Load demo dataset", type="primary", use_container_width=True):
            with st.spinner("Loading synthetic demo records and running the models..."):
                demo_videos, demo_comments = get_demo_data()
                prepare_analysis(demo_videos, demo_comments, 5, "Synthetic demo dataset", demo_mode=True)
            st.success("Demo analysis loaded. All records are simulated and are not real channel analytics.")
            st.rerun()
    with right:
        st.markdown("<div class='muted'>Or open <b>Data & export</b> to upload your own videos CSV and comments CSV.</div>", unsafe_allow_html=True)

    if st.session_state.dataset_ready:
        st.divider()
        st.markdown("### Current analysis snapshot")
        videos = st.session_state.videos
        comments = st.session_state.clustered_comments
        total_views = videos["views"].sum() if "views" in videos else 0
        total_likes = videos["likes"].fillna(0).sum() if "likes" in videos else 0
        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("VIDEOS ANALYZED", fmt_num(len(videos)))
        with c2: metric_card("TOTAL VIEWS", fmt_num(total_views))
        with c3: metric_card("COMMENTS ANALYZED", fmt_num(len(comments)))
        with c4: metric_card("TOTAL LIKES", fmt_num(total_likes))
        st.caption("Choose a page from the left navigation to explore performance, audience segments, anomalies and recommendations.")


elif page == "Video performance":
    show_header("Video performance", "Compare the content that earns attention and interaction.")
    if require_data():
        videos = st.session_state.anomalies.copy()
        views = int(videos["views"].sum())
        likes = int(videos["likes"].fillna(0).sum())
        comments = int(videos["comments_count"].sum())
        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("VIDEOS", fmt_num(len(videos)))
        with c2: metric_card("VIEWS", fmt_num(views))
        with c3: metric_card("LIKES", fmt_num(likes), "Missing public like counts treated as unavailable")
        with c4: metric_card("COMMENTS", fmt_num(comments))

        st.markdown("<div class='section-title'>Engagement by video</div>", unsafe_allow_html=True)
        sort_by = st.selectbox("Rank videos by", ["engagement_rate_pct", "views", "likes", "comments_count"], format_func=lambda x: {
            "engagement_rate_pct": "Engagement rate", "views": "Views", "likes": "Likes", "comments_count": "Comments"
        }[x])
        ranked = videos.sort_values(sort_by, ascending=False).head(15)
        if sort_by == "engagement_rate_pct":
            fig = px.bar(ranked.sort_values(sort_by), x=sort_by, y="title", orientation="h", color_discrete_sequence=["#ff0033"], labels={"engagement_rate_pct": "Engagement rate (%)", "title": "Video"})
        else:
            fig = px.bar(ranked.sort_values(sort_by), x=sort_by, y="title", orientation="h", color_discrete_sequence=["#ff0033"], labels={sort_by: sort_by.replace("_", " ").title(), "title": "Video"})
        fig.update_layout(height=460, margin=dict(l=10, r=10, t=15, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<div class='section-title'>Video library</div>", unsafe_allow_html=True)
        cards = videos.sort_values("views", ascending=False).head(12)
        for start in range(0, len(cards), 3):
            cols = st.columns(3)
            for column, (_, row) in zip(cols, cards.iloc[start:start + 3].iterrows()):
                with column:
                    with st.container(border=True):
                        thumb = str(row.get("thumbnail_url", "") or "")
                        if thumb.startswith("https://"):
                            st.image(thumb, use_container_width=True)
                        else:
                            st.markdown("<div style='background:#eee;border-radius:9px;padding:28px;text-align:center;color:#999'>▶ Video preview</div>", unsafe_allow_html=True)
                        st.markdown(f"**{str(row.get('title', 'Untitled'))[:90]}**")
                        st.caption(f"{fmt_num(row.get('views', 0))} views · {fmt_num(row.get('likes', 0))} likes · {fmt_num(row.get('comments_count', 0))} comments")
                        st.caption(f"Engagement: {fmt_rate(row.get('engagement_rate_pct', 0))}")
                        url = str(row.get("video_url", "") or "")
                        if url.startswith("https://"):
                            st.link_button("Watch on YouTube", url, use_container_width=True)
        display_cols = ["title", "published_at", "views", "likes", "comments_count", "engagement_rate_pct", "anomaly"]
        st.markdown("<div class='section-title'>Video metrics table</div>", unsafe_allow_html=True)
        st.dataframe(videos[[col for col in display_cols if col in videos.columns]], use_container_width=True, hide_index=True)


elif page == "Audience insights":
    show_header("Audience insights", "Discover what viewers are asking, feeling and showing interest in.")
    if require_data():
        comments = st.session_state.clustered_comments.copy()
        if comments.empty:
            st.warning("No comment text is available. Video performance still works; upload a comments CSV or collect public comments to run NLP and K-Means.")
        else:
            if st.session_state.demo_mode:
                st.info("This view uses simulated comments for demonstration. Do not report these as real audience findings.")
            topic_counts = comments["topic"].value_counts().rename_axis("Topic").reset_index(name="Comments")
            sentiment_counts = comments["sentiment"].value_counts().rename_axis("Sentiment").reset_index(name="Comments")
            c1, c2, c3 = st.columns(3)
            with c1: metric_card("COMMENTS ANALYZED", fmt_num(len(comments)))
            with c2: metric_card("POSSIBLE PURCHASE INQUIRIES", f"{comments['purchase_intent'].eq('Possible inquiry').mean():.1%}")
            with c3: metric_card("LANGUAGES / STYLES", fmt_num(comments["language_type"].nunique()))

            left, right = st.columns([1.3, 1])
            with left:
                st.markdown("<div class='section-title'>What viewers talk about</div>", unsafe_allow_html=True)
                fig = px.bar(topic_counts.sort_values("Comments"), x="Comments", y="Topic", orientation="h", color_discrete_sequence=["#ff0033"])
                fig.update_layout(height=390, margin=dict(l=5, r=5, t=10, b=5), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
            with right:
                st.markdown("<div class='section-title'>Comment sentiment</div>", unsafe_allow_html=True)
                fig = px.pie(sentiment_counts, names="Sentiment", values="Comments", hole=0.62, color="Sentiment", color_discrete_map={"Positive": "#16a36a", "Neutral": "#9ca3af", "Negative": "#ff0033"})
                fig.update_layout(height=390, margin=dict(l=5, r=5, t=10, b=5), paper_bgcolor="rgba(0,0,0,0)", legend_title_text="")
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("<div class='section-title'>K-Means audience behaviour segments</div>", unsafe_allow_html=True)
            if st.session_state.silhouette is not None:
                st.caption(f"Silhouette score: {st.session_state.silhouette:.3f} · A rough measure of cluster separation (higher is generally better).")
            else:
                st.caption("Silhouette score is unavailable for this sample size or cluster result.")
            summary = st.session_state.cluster_summary
            if not summary.empty:
                for start in range(0, len(summary), 3):
                    cols = st.columns(3)
                    for column, (_, row) in zip(cols, summary.iloc[start:start + 3].iterrows()):
                        with column:
                            with st.container(border=True):
                                st.markdown(f"**{row['segment']}**")
                                st.metric("Comments in segment", fmt_num(row["comments"]))
                                st.caption(f"{row['share_pct']}% of analyzed comments")
                                st.caption(f"Dominant sentiment: {row['dominant_sentiment']}")
                                st.caption(f"Possible purchase intent: {row['purchase_intent_pct']}%")
                                st.caption(f"Example: “{row['sample_comment']}”")
                st.dataframe(summary, use_container_width=True, hide_index=True)

            st.markdown("<div class='section-title'>Explore analyzed comments</div>", unsafe_allow_html=True)
            topic_filter = st.multiselect("Filter by topic", sorted(comments["topic"].dropna().unique().tolist()), default=[])
            sentiment_filter = st.multiselect("Filter by sentiment", ["Positive", "Neutral", "Negative"], default=[])
            filtered = comments.copy()
            if topic_filter:
                filtered = filtered[filtered["topic"].isin(topic_filter)]
            if sentiment_filter:
                filtered = filtered[filtered["sentiment"].isin(sentiment_filter)]
            show_cols = ["comment_text", "topic", "sentiment", "language_type", "purchase_intent", "cluster_name", "like_count", "reply_count"]
            st.dataframe(filtered[[col for col in show_cols if col in filtered.columns]].head(500), use_container_width=True, hide_index=True)


elif page == "Anomalies & actions":
    show_header("Anomalies & marketing actions", "Find videos outside the normal engagement pattern and decide what to test next.")
    if require_data():
        anomalies = st.session_state.anomalies.copy()
        flagged = anomalies[anomalies["anomaly"] != "Typical"] if not anomalies.empty else anomalies
        c1, c2, c3 = st.columns(3)
        with c1: metric_card("VIDEOS CHECKED", fmt_num(len(anomalies)))
        with c2: metric_card("OUTLIERS FLAGGED", fmt_num(len(flagged)))
        with c3: metric_card("MEDIAN ENGAGEMENT", fmt_rate(anomalies["engagement_rate_pct"].median() if len(anomalies) else 0))
        if len(anomalies) < 6:
            st.info("Isolation Forest is used when at least 6 videos are available. Smaller datasets use a simple robust-rate fallback; collect more videos for a stronger baseline.")
        elif flagged.empty:
            st.info("No unusual videos were flagged in this dataset. This does not mean every video performed equally; it means the model found no strong outlier.")
        else:
            st.markdown("<div class='section-title'>Unusual engagement videos</div>", unsafe_allow_html=True)
            st.dataframe(flagged[[col for col in ["title", "views", "likes", "comments_count", "engagement_rate_pct", "anomaly", "anomaly_score"] if col in flagged.columns]], use_container_width=True, hide_index=True)
        st.markdown("<div class='section-title'>Marketing recommendations</div>", unsafe_allow_html=True)
        for item in st.session_state.recommendations:
            st.markdown(
                f"<div class='recommendation'><div class='recommendation-title'>{item['area']} · {item['priority']} priority</div>"
                f"<div class='recommendation-body'>{item['recommendation']}</div>"
                f"<div class='evidence'>Evidence: {item['evidence']}</div></div>",
                unsafe_allow_html=True,
            )


elif page == "Data & export":
    show_header("Data & export", "Bring your own YouTube CSV files, inspect the dataset and download analysis results.")
    st.markdown("### Upload your own data")
    st.caption("Use one videos CSV and one comments CSV, or a combined CSV containing video fields plus comment_text. See the templates below.")
    with st.expander("Required CSV columns", expanded=True):
        st.markdown(
            "**Videos CSV:** `video_id`, `title`, `views`, `likes`, `comments_count` are the main fields. `video_url`, `published_at`, `thumbnail_url`, `channel_title`, `duration` and `description` are optional.\n\n"
            "**Comments CSV:** `video_id`, `comment_text` are required. `comment_id`, `published_at`, `like_count`, `reply_count` and `author` are optional."
        )
    videos_upload = st.file_uploader("Videos CSV", type=["csv"], key="videos_csv_upload")
    comments_upload = st.file_uploader("Comments CSV (optional)", type=["csv"], key="comments_csv_upload")
    combined_upload = st.file_uploader("Or upload one combined CSV", type=["csv"], key="combined_csv_upload")
    cluster_count_csv = st.slider("K-Means cluster count for uploaded data", 2, 8, 5, key="upload_k")
    if st.button("Load CSV and run analysis", type="primary", use_container_width=True):
        try:
            with st.spinner("Reading CSV files and running analysis..."):
                video_frame, comment_frame = load_csv_uploads(videos_upload, comments_upload, combined_upload)
                prepare_analysis(video_frame, comment_frame, cluster_count_csv, "Uploaded CSV dataset", demo_mode=False)
            st.success(f"Loaded {len(video_frame):,} videos and {len(comment_frame):,} comments.")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not load CSV files: {exc}")

    st.markdown("### Download CSV templates")
    template_videos = pd.DataFrame(columns=VIDEO_COLUMNS)
    template_comments = pd.DataFrame(columns=COMMENT_COLUMNS)
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Download videos template", dataframe_csv_bytes(template_videos), "youtube_videos_template.csv", "text/csv", use_container_width=True)
    with c2:
        st.download_button("Download comments template", dataframe_csv_bytes(template_comments), "youtube_comments_template.csv", "text/csv", use_container_width=True)

    if st.session_state.dataset_ready:
        st.divider()
        st.markdown("### Export current analysis")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button("Videos + performance", dataframe_csv_bytes(st.session_state.anomalies), "youtube_video_performance.csv", "text/csv", use_container_width=True)
        with c2:
            st.download_button("Comments + NLP + clusters", dataframe_csv_bytes(st.session_state.clustered_comments), "youtube_audience_comments.csv", "text/csv", use_container_width=True)
        with c3:
            st.download_button("Cluster summary", dataframe_csv_bytes(st.session_state.cluster_summary), "youtube_audience_segments.csv", "text/csv", use_container_width=True)
        st.markdown("### Data preview")
        st.caption(f"Source: {st.session_state.source_label}")
        if st.session_state.demo_mode:
            st.warning("Synthetic demonstration data. Replace it with real YouTube API or permitted CSV data before reporting project findings.")
        st.dataframe(st.session_state.videos.head(100), use_container_width=True, hide_index=True)
        if st.session_state.comments is not None and not st.session_state.comments.empty:
            st.dataframe(st.session_state.comments.drop(columns=["author"], errors="ignore").head(100), use_container_width=True, hide_index=True)
    st.divider()
    st.markdown("### Important data limitations")
    st.markdown(
        "- The YouTube Data API returns public metadata and public comments; it does not reveal private YouTube Studio analytics such as unique viewers, watch time or audience demographics for arbitrary channels.\n"
        "- The app clusters comment behaviour. It does not prove that multiple comments belong to unique people, nor infer a viewer's age, gender or location.\n"
        "- Public like counts may be hidden or unavailable. Comment threads expose top-level comments; replies are not expanded into a complete reply dataset here.\n"
        "- NLP topic and sentiment labels are lightweight heuristics (including basic Roman Urdu keywords), not a fully trained Urdu language model. Review samples before using the recommendations commercially."
    )

