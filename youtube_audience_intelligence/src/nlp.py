"""Lightweight NLP helpers for English, Roman Urdu and Urdu comments."""

from __future__ import annotations

import re

import pandas as pd
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:  # The built-in fallback keeps CSV/demo analysis usable offline.
    class SentimentIntensityAnalyzer:
        def polarity_scores(self, text: str) -> dict:
            words = set(re.findall(r"[a-z]+", (text or "").lower()))
            positive = len(words.intersection(POSITIVE_WORDS))
            negative = len(words.intersection(NEGATIVE_WORDS))
            compound = (positive - negative) / max(1, positive + negative)
            return {"compound": compound * 0.8}


TOPIC_KEYWORDS = {
    "Price inquiry": ["price", "cost", "rate", "kitna", "kitne", "qeemat", "keemat", "dam", "pkr", "how much", "price please"],
    "Discount / offer": ["discount", "sale", "offer", "coupon", "promo", "deal", "sasta", "kam kar", "kam kardo", "reduced"],
    "Delivery / shipping": ["delivery", "deliver", "shipping", "ship", "parcel", "courier", "tracking", "kitne din", "kab tak", "pohanch", "dispatch"],
    "Availability / location": ["available", "availability", "in stock", "out of stock", "stock", "karachi", "lahore", "islamabad", "where can", "kahan mil", "size available"],
    "Purchase interest": ["buy", "buying", "order", "purchase", "i want", "want this", "interested", "link", "inbox", "dm me", "details send", "book it"],
    "Product / quality": ["quality", "fabric", "material", "cloth", "kapra", "lawn", "cotton", "stitch", "size", "sizing", "fitting", "durable", "review"],
    "Complaint / issue": ["bad", "worst", "poor", "complaint", "disappointed", "fake", "fraud", "scam", "late", "damaged", "not received", "still waiting", "issue", "problem", "refund"],
    "Positive feedback": ["beautiful", "love", "lovely", "amazing", "excellent", "great", "good", "nice", "awesome", "zabardast", "pyara", "pyari", "mashallah", "bohat acha", "best"],
}

POSITIVE_WORDS = {
    "good", "great", "beautiful", "amazing", "excellent", "love", "lovely", "nice", "awesome",
    "perfect", "best", "zabardast", "pyara", "pyari", "acha", "achha", "khoobsurat", "shukriya",
}
NEGATIVE_WORDS = {
    "bad", "worst", "poor", "hate", "disappointed", "fake", "fraud", "scam", "late", "damaged",
    "bekar", "bura", "ghatiya", "mehnga", "mehngi", "masla", "problem", "issue", "not received",
}
PURCHASE_WORDS = {
    "buy", "order", "purchase", "book", "interested", "want this", "i want", "link", "inbox", "dm",
    "order karna", "lena hai", "mujhe chahiye", "details send",
}
ROMAN_URDU_WORDS = {
    "hai", "hain", "bohat", "acha", "achha", "karna", "kardo", "kitna", "kitne", "mein", "mujhe",
    "chahiye", "milay", "milega", "kahan", "kyun", "kab", "nahi", "nahin", "ap", "aap", "bhej",
}


def _contains_any(text: str, words: set[str] | list[str]) -> bool:
    padded = f" {text.lower()} "
    return any(f" {word.lower()} " in padded or word.lower() in text.lower() for word in words)


def detect_language(text: str) -> str:
    if re.search(r"[\u0600-\u06FF]", text or ""):
        return "Urdu"
    tokens = set(re.findall(r"[a-z]+", (text or "").lower()))
    if len(tokens.intersection(ROMAN_URDU_WORDS)) >= 1:
        return "Roman Urdu"
    return "English / Other"


def detect_topic(text: str) -> str:
    clean = (text or "").lower().strip()
    if not clean:
        return "Other / unclear"
    # Complaints are prioritized so a negative delivery issue is not hidden as a delivery question.
    order = [
        "Complaint / issue", "Price inquiry", "Discount / offer", "Delivery / shipping",
        "Availability / location", "Purchase interest", "Product / quality", "Positive feedback",
    ]
    for topic in order:
        if _contains_any(clean, TOPIC_KEYWORDS[topic]):
            return topic
    if "?" in clean or clean.startswith(("what", "why", "when", "where", "how", "can ", "is ", "are ")):
        return "General question"
    return "Other / unclear"


def detect_sentiment(text: str, analyzer: SentimentIntensityAnalyzer) -> str:
    clean = (text or "").lower()
    if not clean.strip():
        return "Neutral"
    positive = _contains_any(clean, POSITIVE_WORDS)
    negative = _contains_any(clean, NEGATIVE_WORDS)
    # Apply simple Roman Urdu / Urdu transliteration clues before English polarity.
    if positive and not negative:
        return "Positive"
    if negative and not positive:
        return "Negative"
    compound = analyzer.polarity_scores(clean)["compound"]
    if compound >= 0.15:
        return "Positive"
    if compound <= -0.15:
        return "Negative"
    return "Neutral"


def analyze_comments(comments: pd.DataFrame) -> pd.DataFrame:
    """Add topic, sentiment, language and purchase-intent columns to comment rows."""
    result = comments.copy()
    if result.empty:
        for column in ["topic", "sentiment", "language_type", "purchase_intent"]:
            result[column] = pd.Series(dtype="object")
        return result

    analyzer = SentimentIntensityAnalyzer()
    text = result["comment_text"].fillna("").astype(str)
    result["topic"] = text.map(detect_topic)
    result["sentiment"] = text.map(lambda item: detect_sentiment(item, analyzer))
    result["language_type"] = text.map(detect_language)
    result["purchase_intent"] = text.map(
        lambda item: "Possible inquiry" if _contains_any(item.lower(), PURCHASE_WORDS) else "Not explicit"
    )
    result["text_length"] = text.str.len()
    return result
