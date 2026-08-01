from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
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
            max_length=256,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR = ROOT_DIR / "models" / "fine_tuned" / "category_model"
DEFAULT_MODEL_NAME = "distilbert-base-uncased"


def load_split(split_name: str) -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / f"tickets_{split_name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Processed split not found: {path}. Please run training/prepare_data.py first."
        )
    return pd.read_csv(path, encoding="utf-8")


def build_label_map(series: pd.Series) -> tuple[dict[str, int], dict[int, str]]:
    labels = sorted(series.dropna().unique())
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def tokenize_batch(examples: dict[str, Any], tokenizer: AutoTokenizer) -> dict[str, Any]:
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    return compute_classification_metrics(eval_pred)


def prepare_dataset(tokenizer: AutoTokenizer, label2id: dict[str, int]) -> dict[str, torch.utils.data.Dataset]:
    dataset_splits: dict[str, torch.utils.data.Dataset] = {}
    for split in ["train", "val"]:
        df = load_split(split)
        df = df[["text", "category"]].dropna().copy()
        df["label"] = df["category"].map(label2id)
        texts = df["text"].tolist()
        labels = df["label"].astype(int).tolist()
        dataset_splits[split] = TextDataset(texts=texts, labels=labels, tokenizer=tokenizer)
    return dataset_splits


def train(args: argparse.Namespace) -> None:
    train_df = load_split("train")
    label2id, id2label = build_label_map(train_df["category"])
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    datasets = prepare_dataset(tokenizer, label2id)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        logging_dir=args.output_dir / "logs",
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved fine-tuned category model to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a category classification model.")
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Base Hugging Face model name or local path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where the fine-tuned category model will be saved.",
    )
    parser.add_argument("--num-train-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-steps", type=int, default=-1, help="Maximum number of training steps/batches.")
    parser.add_argument("--seed", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train(args)
