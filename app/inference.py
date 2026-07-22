from dataclasses import dataclass
import logging
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from app.config import settings
from app.model_handler import model_handler
from app.preprocessing import clean_text
from app.priority_engine import calculate_priority
from app.schemas import PriorityEnum, TicketRequest, TicketResponse

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PredictionResult:
    """Internal container holding the prediction label and score."""

    label: str
    confidence: float


def _predict_single_model(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    labels: dict[int, str],
    threshold: float,
) -> PredictionResult:
    """Helper function performing tokenization, forward pass, softmax, thresholding,

    and label mapping for a single model independently of external singletons.
    """
    # 1. Device direkt vom Modell ableiten (macht die Funktion unabhängig vom Singleton)
    device = next(model.parameters()).device

    # 2. Tokenisierung für Einzelinferenz (padding=False / max_length)
    inputs = tokenizer(
        text,
        max_length=settings.MAX_SEQ_LENGTH,
        padding=False,
        truncation=True,
        return_tensors="pt",
    )

    # Tensors auf das Device des Modells schieben
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # 3. Inferenz ohne Gradientenberechnung
    with torch.inference_mode():
        outputs = model(**inputs)
        # Tensor flachklopfen via squeeze(0)
        probabilities = torch.softmax(outputs.logits.squeeze(0), dim=-1)
        confidence, predicted_class_idx = torch.max(probabilities, dim=-1)

    predicted_idx = int(predicted_class_idx.item())
    confidence_score = round(float(confidence.item()), 4)

    # 4. Thresholding & Label-Lookup
    if confidence_score < threshold:
        predicted_label = "Unknown"
    else:
        predicted_label = labels.get(predicted_idx, "Unknown")

    return PredictionResult(label=predicted_label, confidence=confidence_score)


def predict_ticket(request: TicketRequest) -> TicketResponse:
    """Executes the full end-to-end classification pipeline for an incoming ticket.

    1. Checks model readiness.
    2. Cleans the input text.
    3. Runs category & sentiment inference with configured confidence thresholds.
    4. Calculates business priority deterministically.
    """
    # 1. Fail-Fast: Prüfen, ob alle Modelle initialisiert sind
    if not model_handler.is_healthy():
        raise RuntimeError(
            "Models are not initialized or loaded into memory."
        )

    # Mypy / Type Checker Assertions
    assert model_handler.category_model is not None
    assert model_handler.category_tokenizer is not None
    assert model_handler.category_labels is not None
    assert model_handler.sentiment_model is not None
    assert model_handler.sentiment_tokenizer is not None
    assert model_handler.sentiment_labels is not None

    # 2. Textbereinigung
    cleaned_text = clean_text(request.text)

    # 3. Kategorie-Inferenz mit Threshold
    cat_res = _predict_single_model(
        text=cleaned_text,
        tokenizer=model_handler.category_tokenizer,
        model=model_handler.category_model,
        labels=model_handler.category_labels,
        threshold=getattr(settings, "CATEGORY_CONFIDENCE_THRESHOLD", 0.0),
    )

    # 4. Sentiment-Inferenz mit Threshold
    sent_res = _predict_single_model(
        text=cleaned_text,
        tokenizer=model_handler.sentiment_tokenizer,
        model=model_handler.sentiment_model,
        labels=model_handler.sentiment_labels,
        threshold=getattr(settings, "SENTIMENT_CONFIDENCE_THRESHOLD", 0.0),
    )

    # 5. Business-Priorität berechnen (liefert PriorityEnum)
    priority: PriorityEnum = calculate_priority(
        category=cat_res.label, sentiment=sent_res.label
    )

    # 6. Kompaktes, maschinell lesbares Logging
    # (Nutzt direkt den PriorityEnum bzw. String, kein invalides .value)
    priority_str = priority.value if isinstance(priority, PriorityEnum) else str(priority)
    
    logger.info(
        "Prediction completed | category=%s (%.4f) | sentiment=%s (%.4f) | priority=%s",
        cat_res.label,
        cat_res.confidence,
        sent_res.label,
        sent_res.confidence,
        priority_str,
    )

    # 7. Erstellen der API-Response
    return TicketResponse(
        category=cat_res.label,
        category_confidence=cat_res.confidence,
        sentiment=sent_res.label,
        sentiment_confidence=sent_res.confidence,
        priority=priority,
    )