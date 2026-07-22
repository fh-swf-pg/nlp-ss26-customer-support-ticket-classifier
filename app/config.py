from pathlib import Path
import torch
from pydantic_settings import BaseSettings, SettingsConfigDict

# Basis-Verzeichnis des Projekts bestimmen
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    """
    Zentrale Konfigurationsklasse der Anwendung.
    Werte werden automatisch aus Umgebungsvariablen oder einer .env-Datei geladen.
    """
    
    # -------------------------------------------------------------------------
    # App-Metadaten
    # -------------------------------------------------------------------------
    PROJECT_NAME: str = "Customer Support Ticket Classifier"
    APP_VERSION: str = "1.0.0"
    API_ENV: str = "development"
    
    # -------------------------------------------------------------------------
    # Server-Konfigurationen
    # -------------------------------------------------------------------------
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # -------------------------------------------------------------------------
    # Security & CORS Configuration
    # -------------------------------------------------------------------------
    ALLOWED_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
    ]

    # -------------------------------------------------------------------------
    # Modell-Konfigurationen (Hugging Face Model IDs oder lokale Pfade)
    # -------------------------------------------------------------------------
    # Schalter für den nahtlosen Wechsel zwischen Base & Fine-Tuned
    USE_FINE_TUNED_MODELS: bool = False

    MODELS_DIR: Path = BASE_DIR / "models"
    BASE_MODELS_DIR: Path = MODELS_DIR / "base"
    FINE_TUNED_MODELS_DIR: Path = MODELS_DIR / "fine_tuned"

    @property
    def CATEGORY_MODEL_DIR(self) -> Path:
        if self.USE_FINE_TUNED_MODELS:
            return self.FINE_TUNED_MODELS_DIR / "category_model"
        return self.BASE_MODELS_DIR / "distilbert-base-uncased"

    @property
    def SENTIMENT_MODEL_DIR(self) -> Path:
        if self.USE_FINE_TUNED_MODELS:
            return self.FINE_TUNED_MODELS_DIR / "sentiment_model"
        return self.BASE_MODELS_DIR / "roberta-base"
    
    # -------------------------------------------------------------------------
    # Inferenz-Konfigurationen (Benötigt für inference.py)
    # -------------------------------------------------------------------------
    # Die maximale Token-Länge für die Transformer-Modelle
    MAX_SEQ_LENGTH: int = 512

    # Die minimale Konfidenzschwelle für die Klassifizierung von Kategorien und Sentiment
    CATEGORY_CONFIDENCE_THRESHOLD: float = 0.50
    SENTIMENT_CONFIDENCE_THRESHOLD: float = 0.50

    # Der Gerätetyp für die Modellinferenz
    @property
    def DEVICE(self) -> torch.device:
        return torch.device("cpu")
    
    # -------------------------------------------------------------------------
    # Pydantic Konfiguration
    # -------------------------------------------------------------------------
    # Liest automatisch die .env im Root-Verzeichnis, falls vorhanden
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        # Ignoriert zusätzliche Umgebungsvariablen, die nicht in dieser Klasse definiert sind
        extra="ignore" 
    )

# Instanziierung der Settings zur globalen Verwendung
# (Die Umgebungsvariablen werden genau hier, ein einziges Mal, ausgewertet)
settings = Settings()