from __future__ import annotations

import numpy as np
import evaluate


def compute_classification_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    """Compute common classification metrics from Trainer predictions."""
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=-1)

    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    return {
        "accuracy": float(accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]),
        "f1_weighted": float(
            f1_metric.compute(predictions=preds, references=labels, average="weighted")["f1"]
        ),
    }
