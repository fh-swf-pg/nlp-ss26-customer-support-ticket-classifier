from __future__ import annotations

import re
from typing import Literal

CATEGORY_LABEL_ORDER: list[str] = [
    "System Outage",
    "Security Incident",
    "Network & Connectivity",
    "Hardware Issue",
    "Login & Account Access",
    "Password Reset",
    "Account Management",
    "Billing & Payments",
    "Orders & Subscription",
    "Software Bug",
    "Feature Request",
    "General Inquiry",
]

SENTIMENT_LABEL_ORDER_BY_NUM_LABELS: dict[int, list[str]] = {
    2: ["negative", "positive"],
    3: ["negative", "neutral", "positive"],
    4: ["negative", "neutral", "positive", "angry"],
}

_GENERIC_LABEL_PATTERN = re.compile(r"^LABEL_\d+$", re.IGNORECASE)
_CATEGORY_ALIAS_LOOKUP = {label.lower(): label for label in CATEGORY_LABEL_ORDER}
_SENTIMENT_ALIAS_LOOKUP = {
    "neg": "negative",
    "negative": "negative",
    "neutral": "neutral",
    "pos": "positive",
    "positive": "positive",
    "angry": "angry",
    "anger": "angry",
}


def is_generic_label(label: str) -> bool:
    """Return True when a Hugging Face default label name is detected."""
    return bool(_GENERIC_LABEL_PATTERN.match(label.strip()))


def _extend_labels(base_labels: list[str], total_labels: int, prefix: str) -> list[str]:
    if total_labels <= 0:
        return list(base_labels)

    if total_labels <= len(base_labels):
        return base_labels[:total_labels]

    extended_labels = list(base_labels)
    for index in range(len(base_labels), total_labels):
        extended_labels.append(f"{prefix} {index + 1}")
    return extended_labels


def resolve_label_mapping(
    raw_labels: dict[int, str] | None,
    task: Literal["category", "sentiment"],
    num_labels: int,
) -> dict[int, str]:
    """Return a meaningful label mapping for a classifier model.

    If the model config already contains non-generic labels, those are kept.
    Otherwise, the mapping is derived from the repository's canonical label set.
    """
    if raw_labels:
        normalized_raw_labels = {int(key): str(value) for key, value in raw_labels.items()}
        if not all(is_generic_label(label) for label in normalized_raw_labels.values()):
            return normalized_raw_labels

    if task == "category":
        labels = _extend_labels(CATEGORY_LABEL_ORDER, num_labels, "Category")
    else:
        base_labels = SENTIMENT_LABEL_ORDER_BY_NUM_LABELS.get(num_labels)
        if base_labels is None:
            if num_labels <= 0 or num_labels <= 2:
                base_labels = SENTIMENT_LABEL_ORDER_BY_NUM_LABELS[2]
            elif num_labels == 3:
                base_labels = SENTIMENT_LABEL_ORDER_BY_NUM_LABELS[3]
            else:
                base_labels = SENTIMENT_LABEL_ORDER_BY_NUM_LABELS[4]

        labels = _extend_labels(base_labels, num_labels, "Sentiment")

    return {index: label for index, label in enumerate(labels)}


def normalize_category_label(category: str) -> str:
    """Normalize category labels to the canonical business taxonomy."""
    cleaned = category.strip()
    return _CATEGORY_ALIAS_LOOKUP.get(cleaned.lower(), cleaned)


def normalize_sentiment_label(sentiment: str) -> str:
    """Normalize sentiment labels to the canonical sentiment taxonomy."""
    cleaned = sentiment.strip().lower()
    return _SENTIMENT_ALIAS_LOOKUP.get(cleaned, cleaned)
