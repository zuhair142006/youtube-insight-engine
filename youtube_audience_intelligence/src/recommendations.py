"""Turn measured video/comment patterns into practical next steps."""

from __future__ import annotations

import pandas as pd


def generate_recommendations(videos: pd.DataFrame, comments: pd.DataFrame, anomalies: pd.DataFrame) -> list[dict]:
    recommendations: list[dict] = []
    if videos.empty:
        return recommendations

    if not comments.empty and "topic" in comments.columns:
        topic_counts = comments["topic"].value_counts()
        total = max(1, len(comments))
        for topic, action, reason in [
            ("Price inquiry", "Show prices clearly in the video description or a pinned comment.", "Viewers are asking about price; make the purchase cost easier to find."),
            ("Discount / offer", "Test a time-limited offer and explain its conditions in the video details.", "A noticeable share of comments is focused on discounts and offers."),
            ("Delivery / shipping", "Publish a short delivery FAQ covering locations, delivery times and tracking.", "Delivery questions or concerns are recurring in the comments."),
            ("Availability / location", "Add a clear availability / location note and keep stock information current.", "Viewers are asking where or whether the product is available."),
            ("Purchase interest", "Use a clear call to action and link the product or next step in the description.", "Comments contain direct signals of product or purchase interest."),
            ("Product / quality", "Create a close-up demonstration covering material, sizing, features or quality.", "Viewers want more detail before making a decision."),
            ("Complaint / issue", "Review the recurring issue and respond with a visible service update.", "Some comments describe problems that may affect trust."),
        ]:
            count = int(topic_counts.get(topic, 0))
            share = count / total
            if count >= 2 and share >= 0.06:
                recommendations.append({"priority": "High" if share >= 0.15 or topic == "Complaint / issue" else "Medium", "area": topic, "recommendation": action, "evidence": f"{count:,} of {total:,} analyzed comments ({share:.1%}) were tagged as {topic.lower()}."})

        negative_share = comments["sentiment"].eq("Negative").mean() if "sentiment" in comments else 0
        if negative_share >= 0.20:
            recommendations.append({
                "priority": "High", "area": "Audience sentiment",
                "recommendation": "Review the most-liked negative comments first and publish a direct response to the main concern.",
                "evidence": f"{negative_share:.1%} of analyzed comments were tagged negative by the NLP rules / sentiment model.",
            })

    if "engagement_rate_pct" in anomalies and len(anomalies):
        median_rate = float(anomalies["engagement_rate_pct"].median())
        high = anomalies[anomalies["anomaly"] == "Unusually high engagement"]
        low = anomalies[anomalies["anomaly"] == "Unusually low engagement"]
        if not high.empty:
            titles = ", ".join(high["title"].astype(str).head(2).tolist())
            recommendations.append({
                "priority": "High", "area": "High-performing outlier",
                "recommendation": "Compare the opening hook, topic, thumbnail and format of the outlier videos; test a follow-up with the same winning pattern.",
                "evidence": f"{len(high)} video(s) were flagged as unusually high engagement. Example: {titles}.",
            })
        if not low.empty:
            titles = ", ".join(low["title"].astype(str).head(2).tolist())
            recommendations.append({
                "priority": "Medium", "area": "Low-performing outlier",
                "recommendation": "Review the opening seconds, title, thumbnail and audience fit before repeating this content format.",
                "evidence": f"{len(low)} video(s) were flagged as unusually low engagement. Example: {titles}.",
            })
        if median_rate > 0:
            recommendations.append({
                "priority": "Low", "area": "Engagement baseline",
                "recommendation": "Track engagement rate alongside raw views so that smaller videos can be compared more fairly.",
                "evidence": f"The median engagement rate in this dataset is {median_rate:.2f}%.",
            })

    if comments.empty:
        recommendations.append({
            "priority": "Medium", "area": "Comment coverage",
            "recommendation": "Upload a comments CSV or analyze videos with public comments enabled to unlock audience segmentation.",
            "evidence": "No comment text is available in the current dataset; comment needs and clusters cannot be inferred.",
        })

    if not recommendations:
        recommendations.append({
            "priority": "Low", "area": "Continue testing",
            "recommendation": "Collect more videos and comments across comparable content formats, then compare results over time.",
            "evidence": "The current dataset does not yet show a strong repeated audience or performance pattern.",
        })
    return recommendations
