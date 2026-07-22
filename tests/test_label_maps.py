from app.label_maps import normalize_category_label, resolve_label_mapping


def test_resolve_label_mapping_uses_canonical_category_labels() -> None:
    labels = resolve_label_mapping(
        raw_labels={0: "LABEL_0", 1: "LABEL_1"},
        task="category",
        num_labels=2,
    )

    assert labels == {0: "System Outage", 1: "Security Incident"}


def test_resolve_label_mapping_uses_canonical_sentiment_labels() -> None:
    labels = resolve_label_mapping(
        raw_labels={0: "LABEL_0", 1: "LABEL_1"},
        task="sentiment",
        num_labels=2,
    )

    assert labels == {0: "negative", 1: "positive"}


def test_normalize_category_label_is_case_insensitive() -> None:
    assert normalize_category_label("system outage") == "System Outage"