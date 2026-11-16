"""K-Means audience-behaviour segmentation and Isolation Forest outlier detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def cluster_comments(comments: pd.DataFrame, n_clusters: int = 5) -> tuple[pd.DataFrame, pd.DataFrame, float | None]:
    """Cluster comment behaviour using TF-IDF text plus interaction signals."""
    if comments.empty:
        return comments.copy(), pd.DataFrame(), None

    result = comments.copy().reset_index(drop=True)
    for col, default in [("topic", "Other / unclear"), ("sentiment", "Neutral"), ("language_type", "English / Other"), ("purchase_intent", "Not explicit")]:
        if col not in result.columns:
            result[col] = default

    text = result["comment_text"].fillna("").astype(str)
    # Add NLP labels as tokens so the model groups similar needs as well as similar wording.
    documents = (
        text + " " +
        "topic_" + result["topic"].astype(str).str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True) + " " +
        "sentiment_" + result["sentiment"].astype(str).str.lower() + " " +
        "language_" + result["language_type"].astype(str).str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True) + " " +
        "purchase_" + result["purchase_intent"].astype(str).str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True)
    )
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=450, min_df=1, sublinear_tf=True)
    text_matrix = vectorizer.fit_transform(documents)

    like_counts = result["like_count"] if "like_count" in result.columns else pd.Series(0, index=result.index)
    reply_counts = result["reply_count"] if "reply_count" in result.columns else pd.Series(0, index=result.index)
    numeric = pd.DataFrame({
        "text_length": text.str.len().astype(float),
        "comment_likes": pd.to_numeric(like_counts, errors="coerce").fillna(0).astype(float),
        "reply_count": pd.to_numeric(reply_counts, errors="coerce").fillna(0).astype(float),
        "has_question": text.str.contains(r"\?|\bwhat\b|\bhow\b|\bwhere\b|\bwhen\b|\bcan\b", case=False, regex=True).astype(float),
        "purchase_intent_flag": result["purchase_intent"].eq("Possible inquiry").astype(float),
    })
    numeric_matrix = StandardScaler(with_mean=False).fit_transform(numeric)
    features = hstack([text_matrix, csr_matrix(numeric_matrix)], format="csr")

    count = len(result)
    if count < 3:
        result["cluster"] = 0
        result["cluster_name"] = "Small sample"
        return result, summarize_clusters(result), None

    k = max(2, min(int(n_clusters), count - 1))
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    result["cluster"] = model.fit_predict(features)
    score = None
    if k > 1 and count > k and result["cluster"].nunique() > 1:
        try:
            # Sampling avoids expensive silhouette scoring on large comment datasets.
            sample_size = min(count, 1500)
            score = float(silhouette_score(features, result["cluster"], sample_size=sample_size, random_state=42))
        except Exception:
            score = None
    summary = summarize_clusters(result)
    name_map = {}
    for _, row in summary.iterrows():
        name_map[int(row["cluster"])] = f"{row['top_topic']} segment"
    result["cluster_name"] = result["cluster"].map(name_map).fillna("Audience segment")
    summary = summarize_clusters(result)
    return result, summary, score


def summarize_clusters(clustered: pd.DataFrame) -> pd.DataFrame:
    if clustered.empty or "cluster" not in clustered.columns:
        return pd.DataFrame()
    rows = []
    for cluster_id, group in clustered.groupby("cluster", sort=True):
        topic_counts = group["topic"].value_counts() if "topic" in group else pd.Series(dtype=int)
        sentiment_counts = group["sentiment"].value_counts() if "sentiment" in group else pd.Series(dtype=int)
        top_topic = str(topic_counts.index[0]) if len(topic_counts) else "Mixed interests"
        top_topic_group = group[group["topic"] == top_topic] if "topic" in group else group
        sample_comment = str(top_topic_group["comment_text"].iloc[0] if len(top_topic_group) else group["comment_text"].iloc[0])[:140]
        rows.append({
            "cluster": int(cluster_id),
            "segment": f"{top_topic} segment",
            "comments": int(len(group)),
            "share_pct": round(100 * len(group) / len(clustered), 1),
            "top_topic": top_topic,
            "dominant_sentiment": str(sentiment_counts.index[0]) if len(sentiment_counts) else "Unknown",
            "purchase_intent_pct": round(100 * group["purchase_intent"].eq("Possible inquiry").mean(), 1) if "purchase_intent" in group else 0.0,
            "sample_comment": sample_comment,
        })
    return pd.DataFrame(rows).sort_values("comments", ascending=False).reset_index(drop=True)


def detect_video_anomalies(videos: pd.DataFrame) -> pd.DataFrame:
    """Use Isolation Forest on video scale and engagement-rate features."""
    if videos.empty:
        return videos.copy()
    result = videos.copy()
    for col in ["views", "likes", "comments_count"]:
        if col not in result.columns:
            result[col] = 0
        result[col] = pd.to_numeric(result[col], errors="coerce")
    result["views"] = result["views"].fillna(0).clip(lower=0)
    result["comments_count"] = result["comments_count"].fillna(0).clip(lower=0)
    result["likes"] = result["likes"].fillna(0).clip(lower=0)
    safe_views = result["views"].replace(0, np.nan)
    result["like_rate_pct"] = (result["likes"] / safe_views * 100).replace([np.inf, -np.inf], np.nan).fillna(0)
    result["comment_rate_pct"] = (result["comments_count"] / safe_views * 100).replace([np.inf, -np.inf], np.nan).fillna(0)
    result["engagement_rate_pct"] = ((result["likes"] + result["comments_count"]) / safe_views * 100).replace([np.inf, -np.inf], np.nan).fillna(0)
    result["anomaly"] = "Typical"
    result["anomaly_score"] = np.nan

    # Isolation Forest needs a small sample of multiple videos to learn a baseline.
    if len(result) >= 6:
        features = pd.DataFrame({
            "log_views": np.log1p(result["views"]),
            "like_rate": result["like_rate_pct"],
            "comment_rate": result["comment_rate_pct"],
            "engagement_rate": result["engagement_rate_pct"],
        }).replace([np.inf, -np.inf], 0).fillna(0)
        model = IsolationForest(n_estimators=200, contamination=min(0.20, max(0.08, 2 / len(result))), random_state=42)
        prediction = model.fit_predict(features)
        scores = model.decision_function(features)
        result["anomaly_score"] = scores
        outliers = prediction == -1
        median = result["engagement_rate_pct"].median()
        result.loc[outliers & (result["engagement_rate_pct"] > median), "anomaly"] = "Unusually high engagement"
        result.loc[outliers & (result["engagement_rate_pct"] < median), "anomaly"] = "Unusually low engagement"
        result.loc[outliers & (result["engagement_rate_pct"] == median), "anomaly"] = "Unusual metric pattern — review"
    elif len(result) >= 3:
        # Small-sample fallback: robust median absolute deviation on engagement rate.
        values = result["engagement_rate_pct"].to_numpy(dtype=float)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        if mad > 0:
            robust_z = 0.6745 * (values - median) / mad
            high = robust_z >= 3.5
            low = robust_z <= -3.5
            result.loc[high, "anomaly"] = "Unusually high engagement"
            result.loc[low, "anomaly"] = "Unusually low engagement"
    return result
