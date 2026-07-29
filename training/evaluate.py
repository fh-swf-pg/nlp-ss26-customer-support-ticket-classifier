from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer
import evaluate

try:
    from training.metrics import compute_classification_metrics
except ImportError:
    from metrics import compute_classification_metrics


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer: AutoTokenizer) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
CATEGORY_MODEL_DIR = ROOT_DIR / "models" / "fine_tuned" / "category_model"
SENTIMENT_MODEL_DIR = ROOT_DIR / "models" / "fine_tuned" / "sentiment_model"


def load_split(split_name: str) -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / f"tickets_{split_name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Processed split not found: {path}. Please run training/prepare_data.py first."
        )
    return pd.read_csv(path, encoding="utf-8")


def evaluate_task(model_dir: Path, label_column: str) -> dict[str, float]:
    df = load_split("test")[["text", label_column]].dropna().copy()

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)

    label2id = getattr(model.config, "label2id", {}) or {}
    labels = [label2id[label] for label in df[label_column].tolist()]
    dataset = TextDataset(texts=df["text"].tolist(), labels=labels, tokenizer=tokenizer)

    trainer = Trainer(model=model, tokenizer=tokenizer)
    predictions = trainer.predict(dataset)

    return compute_classification_metrics((predictions.predictions, predictions.label_ids))


def main() -> None:
    print("Evaluating category model...")
    category_metrics = evaluate_task(CATEGORY_MODEL_DIR, "category")
    print("Category metrics:", category_metrics)

    print("Evaluating sentiment model...")
    sentiment_metrics = evaluate_task(SENTIMENT_MODEL_DIR, "sentiment")
    print("Sentiment metrics:", sentiment_metrics)


if __name__ == "__main__":
    main()
