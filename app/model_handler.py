import logging
from pathlib import Path
from typing import Literal, Optional

from torch import device
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from app.config import settings
from app.label_maps import resolve_label_mapping

logger = logging.getLogger(__name__)


class ModelHandler:
    """Loads and manages locally stored Hugging Face models used for inference.

    Models are initialized once during FastAPI startup and reused for all incoming requests.
    """

    def __init__(self) -> None:
        self.device: device = settings.DEVICE

        self._category_tokenizer: Optional[PreTrainedTokenizerBase] = None
        self._category_model: Optional[PreTrainedModel] = None
        self._category_labels: Optional[dict[int, str]] = None

        self._sentiment_tokenizer: Optional[PreTrainedTokenizerBase] = None
        self._sentiment_model: Optional[PreTrainedModel] = None
        self._sentiment_labels: Optional[dict[int, str]] = None

    def _load_single_model(
        self, model_path: Path, task: Literal["category", "sentiment"]
    ) -> tuple[PreTrainedTokenizerBase, PreTrainedModel, dict[int, str]]:
        """Private helper method to load a tokenizer, model, and labels strictly from

        a local directory.
        """
        # 1. Check if path is a valid directory
        if not model_path.is_dir():
            raise FileNotFoundError(
                f"Model directory not found: {model_path}. "
                "Please run 'python -m scripts.download_models' first!"
            )

        # 2. Check for Hugging Face configuration file
        config_file = model_path / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(
                f"No Hugging Face model found in {model_path}. Missing 'config.json'."
            )

        # 3. Load Tokenizer & Model strictly offline
        logger.info("Loading tokenizer from %s...", model_path)
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True
        )

        logger.info("Loading model weights from %s...", model_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path), local_files_only=True
        )
        model.to(self.device)
        model.eval()

        # 4. Extract and validate id2label mapping
        raw_labels = getattr(model.config, "id2label", None)
        num_labels = int(getattr(model.config, "num_labels", 0) or 0)
        labels = resolve_label_mapping(raw_labels, task=task, num_labels=num_labels)
        model.config.id2label = labels
        model.config.label2id = {label: idx for idx, label in labels.items()}

        return tokenizer, model, labels

    def load_models(self) -> None:
        """Loads both category and sentiment models into memory from local paths."""
        logger.info("Initializing local model loading from disk...")

        try:
            # 1. Load Category Model
            logger.info("Initializing Category Model from %s", settings.CATEGORY_MODEL_DIR)
            (
                self._category_tokenizer,
                self._category_model,
                self._category_labels,
            ) = self._load_single_model(settings.CATEGORY_MODEL_DIR, task="category")
            logger.info("Category labels loaded successfully: %s", self._category_labels)

            # 2. Load Sentiment Model
            logger.info("Initializing Sentiment Model from %s", settings.SENTIMENT_MODEL_DIR)
            (
                self._sentiment_tokenizer,
                self._sentiment_model,
                self._sentiment_labels,
            ) = self._load_single_model(settings.SENTIMENT_MODEL_DIR, task="sentiment")
            logger.info("Sentiment labels loaded successfully: %s", self._sentiment_labels)

            logger.info("All local models successfully initialized.")

        except Exception:
            logger.exception("Failed to load local models from disk.")
            raise

    def unload_models(self) -> None:
        """Frees memory by unreferencing all loaded models, tokenizers, and labels."""
        logger.info("Unloading models from memory...")
        self._category_model = None
        self._category_tokenizer = None
        self._category_labels = None

        self._sentiment_model = None
        self._sentiment_tokenizer = None
        self._sentiment_labels = None
        logger.info("All models successfully unloaded.")

    def is_healthy(self) -> bool:
        """Checks if all local models, tokenizers, and label mappings are ready."""
        return all([
            self._category_tokenizer is not None,
            self._category_model is not None,
            self._category_labels is not None,
            self._sentiment_tokenizer is not None,
            self._sentiment_model is not None,
            self._sentiment_labels is not None,
        ])

    # -------------------------------------------------------------------------
    # Properties (Pythonic Accessors)
    # -------------------------------------------------------------------------
    @property
    def category_model(self) -> Optional[PreTrainedModel]:
        return self._category_model

    @property
    def category_tokenizer(self) -> Optional[PreTrainedTokenizerBase]:
        return self._category_tokenizer

    @property
    def category_labels(self) -> Optional[dict[int, str]]:
        return self._category_labels

    @property
    def sentiment_model(self) -> Optional[PreTrainedModel]:
        return self._sentiment_model

    @property
    def sentiment_tokenizer(self) -> Optional[PreTrainedTokenizerBase]:
        return self._sentiment_tokenizer

    @property
    def sentiment_labels(self) -> Optional[dict[int, str]]:
        return self._sentiment_labels


# Global Singleton Instance
model_handler = ModelHandler()