from pathlib import Path

from transformers import AutoModel, AutoTokenizer

# Base models
CATEGORY_MODEL_NAME = "distilbert-base-uncased"
SENTIMENT_MODEL_NAME = "roberta-base"

# Directory structure
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = PROJECT_ROOT / "models"
BASE_MODELS_DIR = MODELS_DIR / "base"

CATEGORY_MODEL_DIR = BASE_MODELS_DIR / CATEGORY_MODEL_NAME
SENTIMENT_MODEL_DIR = BASE_MODELS_DIR / SENTIMENT_MODEL_NAME


def download_model(model_name: str, output_dir: Path) -> None:
    """
    Downloads a Hugging Face base model and tokenizer
    and stores them locally.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)

    print(f"✓ Saved to: {output_dir}\n")


def main() -> None:
    download_model(
        CATEGORY_MODEL_NAME,
        CATEGORY_MODEL_DIR,
    )

    download_model(
        SENTIMENT_MODEL_NAME,
        SENTIMENT_MODEL_DIR,
    )

    print("All base models downloaded successfully.")


if __name__ == "__main__":
    main()