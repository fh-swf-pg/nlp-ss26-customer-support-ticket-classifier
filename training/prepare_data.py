from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

CATEGORY_KEYWORD_MAP: dict[str, list[str]] = {
    "System Outage": [
        "system outage",
        "service outage",
        "outage",
        "disruption",
        "downtime",
        "unavailable",
        "offline",
        "interruption",
        "service down",
    ],
    "Security Incident": [
        "security",
        "breach",
        "data breach",
        "unauthorized",
        "unauthorised",
        "suspicious",
        "gdpr",
        "hipaa",
        "attack",
        "vulnerability",
        "compromised",
    ],
    "Network & Connectivity": [
        "network",
        "connectivity",
        "vpn",
        "router",
        "internet",
        "wifi",
        "dns",
        "latency",
        "timeout",
        "packet loss",
        "connection",
        "disconnect",
        "bandwidth",
    ],
    "Hardware Issue": [
        "hardware",
        "printer",
        "device",
        "laptop",
        "server",
        "scanner",
        "battery",
        "keyboard",
        "monitor",
        "screen",
        "disk",
        "drive",
        "camera",
        "cpu",
        "physical",
        "unit",
    ],
    "Login & Account Access": [
        "login",
        "sign in",
        "sign-in",
        "sign in",
        "log in",
        "signin",
        "account access",
        "access denied",
        "locked out",
        "account blocked",
        "two-factor",
        "two factor",
        "authentication",
    ],
    "Password Reset": [
        "password reset",
        "forgot password",
        "reset password",
        "password change",
        "password expired",
        "password issue",
    ],
    "Account Management": [
        "profile",
        "account settings",
        "update account",
        "change account",
        "account information",
        "account details",
        "user account",
        "manage account",
        "account owner",
    ],
    "Billing & Payments": [
        "billing",
        "payment",
        "invoice",
        "charge",
        "refund",
        "subscription fee",
        "receipt",
        "payment method",
        "billing cycle",
        "payment failed",
    ],
    "Orders & Subscription": [
        "order",
        "subscription",
        "delivery",
        "shipment",
        "order status",
        "renewal",
        "cancel subscription",
        "order number",
        "shipping",
    ],
    "Software Bug": [
        "bug",
        "error",
        "crash",
        "failure",
        "fault",
        "exception",
        "stack trace",
        "unexpected behavior",
        "not working",
        "broken",
        "debug",
    ],
    "Feature Request": [
        "feature request",
        "new feature",
        "enhancement",
        "improvement",
        "suggestion",
        "wish",
        "feature",
    ],
}

SENTIMENT_PRIORITY_MAP: dict[str, str] = {
    "critical": "negative",
    "high": "negative",
    "medium": "neutral",
    "moderate": "neutral",
    "low": "positive",
    "very_low": "positive",
    "very low": "positive",
    "verylow": "positive",
    "low/medium": "neutral",
}

MISSING_CATEGORY = "General Inquiry"
MISSING_SENTIMENT = "neutral"


def _read_csv_file(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str, encoding="utf-8", keep_default_na=False, on_bad_lines="skip")
    except UnicodeDecodeError:
        return pd.read_csv(path, dtype=str, encoding="latin-1", keep_default_na=False, on_bad_lines="skip")


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _collect_text_fields(row: pd.Series) -> str:
    subject = _safe_text(row.get("subject", ""))
    body = _safe_text(row.get("body", ""))
    if subject and body:
        return f"{subject}\n\n{body}"
    return subject or body


def _match_category(tokens: Iterable[str]) -> str | None:
    for category, patterns in CATEGORY_KEYWORD_MAP.items():
        for token in tokens:
            candidate = _safe_text(token).lower()
            if not candidate:
                continue
            if any(pattern in candidate for pattern in patterns):
                return category
    return None


def _infer_category(row: pd.Series) -> str:
    tokens = []
    tokens.extend(str(row.get("queue", "")).split(" "))
    tokens.extend(str(row.get("type", "")).split(" "))
    tokens.extend(str(row.get("business_type", "")).split(" "))
    tags = [row.get(f"tag_{i}", "") for i in range(1, 10)]
    tokens.extend(tags)
    tokens.append(_safe_text(row.get("subject", "")))
    tokens.append(_safe_text(row.get("body", "")))

    exact = _match_category(tags)
    if exact:
        return exact

    candidate = _match_category([row.get("queue", ""), row.get("type", ""), row.get("business_type", "")])
    if candidate:
        return candidate

    candidate = _match_category([row.get("subject", ""), row.get("body", "")])
    if candidate:
        return candidate

    queue = _safe_text(row.get("queue", "")).lower()
    if "billing" in queue or "payment" in queue or "invoice" in queue:
        return "Billing & Payments"
    if "subscription" in queue or "order" in queue or "delivery" in queue:
        return "Orders & Subscription"
    if "security" in queue or "gdpr" in queue or "hipaa" in queue:
        return "Security Incident"
    if "hardware" in queue or "printer" in queue or "device" in queue:
        return "Hardware Issue"
    if "network" in queue or "connectivity" in queue or "vpn" in queue:
        return "Network & Connectivity"
    if "login" in queue or "account access" in queue or "access" in queue:
        return "Login & Account Access"

    return MISSING_CATEGORY


def _infer_sentiment(row: pd.Series) -> str:
    priority = _safe_text(row.get("priority", "")).lower()
    normalized = re.sub(r"[\s_-]+", " ", priority).strip()
    if normalized in SENTIMENT_PRIORITY_MAP:
        return SENTIMENT_PRIORITY_MAP[normalized]

    text = _safe_text(row.get("subject", "")) + " " + _safe_text(row.get("body", ""))
    lower_text = text.lower()
    if any(token in lower_text for token in ["urgent", "critical", "failure", "error", "problem", "crash", "unable", "blocked", "delay", "issue"]):
        return "negative"
    if any(token in lower_text for token in ["thank", "appreciate", "good", "great", "helpful", "info"]):
        return "positive"
    return MISSING_SENTIMENT


def _prepare_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.rename(
        columns={
            "text": "text",
            "description": "body",
            "request": "body",
        }
    )
    raw["subject"] = raw.get("subject", "")
    raw["body"] = raw.get("body", "")
    raw["queue"] = raw.get("queue", "")
    raw["priority"] = raw.get("priority", "")
    raw["type"] = raw.get("type", "")
    raw["business_type"] = raw.get("business_type", "")

    raw["text"] = raw.apply(_collect_text_fields, axis=1).astype(str)
    raw["text"] = raw["text"].apply(_normalize_text)
    raw = raw[raw["text"].astype(bool)].copy()

    raw["category"] = raw.apply(_infer_category, axis=1)
    raw["sentiment"] = raw.apply(_infer_sentiment, axis=1)
    raw["language"] = raw.get("language", "en").replace("", "en")

    return raw[["text", "category", "sentiment", "priority", "language"]]


def load_raw_datasets() -> pd.DataFrame:
    csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No raw CSV files found in {RAW_DATA_DIR}")

    frames: list[pd.DataFrame] = []
    for csv_path in csv_files:
        df = _read_csv_file(csv_path)
        df["source_file"] = csv_path.name
        frames.append(df)

    return pd.concat(frames, ignore_index=True, sort=False)


def split_dataset(data: pd.DataFrame, seed: int = 42) -> dict[str, pd.DataFrame]:
    data = data.reset_index(drop=True)
    if len(data) < 10:
        raise ValueError("Not enough data available to create train/validation/test splits.")

    stratify = data["category"] if data["category"].nunique() > 1 else None
    if stratify is not None and data["category"].value_counts().min() < 2:
        stratify = None

    train, temp = train_test_split(
        data,
        test_size=0.20,
        random_state=seed,
        stratify=stratify,
    )

    stratify_temp = temp["category"] if stratify is not None else None
    if stratify_temp is not None and stratify_temp.value_counts().min() < 2:
        stratify_temp = None

    val, test = train_test_split(
        temp,
        test_size=0.50,
        random_state=seed,
        stratify=stratify_temp,
    )

    return {"train": train.reset_index(drop=True), "val": val.reset_index(drop=True), "test": test.reset_index(drop=True)}


def save_processed_splits(splits: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    full_path = output_dir / "tickets_processed.csv"
    pd.concat(splits.values(), ignore_index=True).to_csv(full_path, index=False, encoding="utf-8")
    for name, frame in splits.items():
        frame.to_csv(output_dir / f"tickets_{name}.csv", index=False, encoding="utf-8")


def main() -> None:
    print("Loading raw data from", RAW_DATA_DIR)
    raw = load_raw_datasets()
    print(f"Loaded {len(raw):,} rows from raw CSV files.")

    processed = _prepare_dataframe(raw)
    print(f"Prepared dataset with {len(processed):,} tickets.")

    splits = split_dataset(processed)
    save_processed_splits(splits, PROCESSED_DATA_DIR)

    print("Saved processed splits to", PROCESSED_DATA_DIR)
    print("Files:")
    for split in ["train", "val", "test"]:
        print(" -", PROCESSED_DATA_DIR / f"tickets_{split}.csv")


if __name__ == "__main__":
    main()
