# 🎫 Customer Support Ticket Classifier

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU-ee4c2c)
![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow)
![License](https://img.shields.io/badge/license-MIT-green)

Ein modularer, produktionsnaher NLP-Microservice zur automatischen Klassifizierung und Priorisierung von Kundensupport-Tickets. Das System nutzt feingetunte Transformer-Modelle (**DistilBERT** für IT-Kategorien und **RoBERTa** für Sentiment-Analysen), leitet daraus dynamisch eine Priorität ab und stellt die Inferenz über eine performante **FastAPI**-Schnittstelle bereit.
---

## ✨ Hauptmerkmale

- **Dual-Model Inference:** Parallelisierung / Ausführung von zwei spezialisierten Transformer-Modellen (Kategorie & Sentiment).
- **Regelbasierte Priority Engine:** Dynamische Berechnung der Ticket-Dringlichkeit basierend auf Kategorie-Gefährdung und Frustrationsgrad.
- **Production-Ready & Lightweight:** Nutzt `safetensors` für schnelles Laden und CPU-optimiertes PyTorch für minimale Container-Größen.
- **Clean Architecture:** Klare Trennung von Training/Entwicklung (`requirements-dev.txt`) und schlanker Production-Inferenz (`requirements.txt`).
- **Full Test Coverage:** Automatisierte Unit- und Integrationstests via `pytest` inkl. GitHub Actions CI/CD.

---

## 🏗️ MLOps- & System-Architektur

Das Projekt trennt strikt zwischen **Lokaler Modellentwicklung/Training**, **Modell-Publishing** und **Inferenz-Deployment**:

```text
+---------------------------+
|     GitHub Repository     |
|---------------------------|
| app/                      |
| training/                 |
| scripts/                  |
| notebooks/                |
| tests/                    |
| docs/                     |
| Dockerfile                |
| requirements*.txt         |
+-------------+-------------+
              |
              v
+---------------------------+
| Hugging Face Spaces       |
|---------------------------|
| FastAPI Engine            |
| (Lädt Modelle beim Start) |
+-------------+-------------+
              |
              v
+---------------------------+
| Hugging Face Model Hub    |
|---------------------------|
| category-model            |
| sentiment-model           |
+---------------------------+

---

## 🏗️ System-Architektur

Das Projekt trennt strikt zwischen **Lokaler Modellentwicklung/Training** und schlankem **Inferenz-Deployment**:

```text
        Eingehendes Ticket (Text)
                   │
                   ▼
┌────────────────────────────────────────┐
│             FastAPI Engine             │
│  1. Text-Preprocessing (preprocessing) │
│  2. Tokenisierung & Modell-Inferenz    │
└──────────────────┬─────────────────────┘
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
┌──────────────────┐┌──────────────────┐
│ Category Model   ││ Sentiment Model  │
│  (DistilBERT)    ││   (RoBERTa)      │
└────────┬─────────┘└────────┬─────────┘
         │                   │
         └─────────┬─────────┘
                   ▼
┌────────────────────────────────────────┐
│            Priority Engine             │
│   (Berechnet Priorität aus Kategorie   │
│            + Sentiment)                │
└──────────────────┬─────────────────────┘
                   ▼
┌────────────────────────────────────────┐
│             JSON Response              │
│ - Category & Confidence                │
│ - Sentiment & Confidence               │
│ - Priority                             │
└────────────────────────────────────────┘
```

---

## 🎯 Ticket-Kategorien

Das Modell klassifiziert eingehende Kundensupport-Tickets in eine der folgenden Kategorien:

| Kategorie | Typische Beispiele | Standardpriorität |
|-----------|--------------------|-------------------|
| 🔐 Login & Account Access | Login fehlgeschlagen, Konto gesperrt | Medium |
| 🔑 Password Reset | Passwort vergessen | Low |
| 👤 Account Management | Benutzerprofil, Kontoeinstellungen | Low |
| 💳 Billing & Payments | Rechnung, Zahlung, Erstattung | Low |
| 🛒 Orders & Subscription | Bestellung, Abo, Verlängerung | Medium |
| 🐞 Software Bug | Fehlermeldungen, Abstürze | Medium |
| 🌐 Network & Connectivity | Verbindungsprobleme | High |
| 💻 Hardware Issue | Defekte Geräte | High |
| 🚨 System Outage | Service nicht verfügbar | Critical |
| 🔒 Security Incident | Verdächtige Anmeldung, Datenleck | Critical |
| 💡 Feature Request | Wunsch nach neuer Funktion | Low |
| ❓ General Inquiry | Allgemeine Fragen | Low |

---

## 🚦 Prioritätsberechnung

Die Priorität wird **nicht direkt vom Machine-Learning-Modell vorhergesagt**, sondern von der **Business-Logik (`priority_engine.py`)** berechnet.

Dazu kombiniert die Anwendung die vorhergesagte **Ticketkategorie** mit dem **Sentiment**, um Tickets mit hohem Eskalationspotenzial automatisch zu priorisieren.

| Kategorie | Sentiment | Berechnete Priorität |
|-----------|-----------|----------------------|
| **System Outage** | Beliebig | 🔴 **Critical** |
| **Security Incident** | Beliebig | 🔴 **Critical** |
| **Hardware Issue** | Negative oder Angry | 🔴 **High** |
| **Network & Connectivity** | Negative oder Angry | 🔴 **High** |
| **Software Bug** | Negative oder Angry | 🟡 **Medium** |
| **Login & Account Access** | Negative oder Angry | 🟡 **Medium** |
| **Billing & Payments** | Negative oder Angry | 🟡 **Medium** |
| **Orders & Subscription** | Negative oder Angry | 🟡 **Medium** |
| **Alle übrigen Kategorien** | Neutral oder Positive | 🟢 **Low** |

> **Hinweis:** Die Prioritätsregeln sind vollständig konfigurierbar und können jederzeit erweitert oder an die Anforderungen eines Unternehmens angepasst werden.
---

## 📁 Projektstruktur

```text
customer-support-ticket-classifier/
│
├── .github/
│   └── workflows/
│       └── python-tests.yml        # CI/CD Pipeline
│
├── app/                            # FastAPI Backend (Production Inferenz)
│   ├── __init__.py
│   ├── main.py                     # API REST-Endpunkte (/health, /predict)
│   ├── config.py                   # App-Konfiguration & Umgebungsvariablen
│   ├── schemas.py                  # Pydantic Schemas für Request/Response
│   ├── model_handler.py            # Lädt Modelle aus models/ in den Speicher
│   ├── inference.py                # Führt Modellvorhersagen aus
│   ├── preprocessing.py            # Textbereinigung & Normalisierung
│   └── priority_engine.py          # Business-Logik zur Prioritätsberechnung
│
├── training/                       # Skripte für lokales Fine-Tuning
│   ├── prepare_data.py             # Datensatz-Bereinigung & Split (Train/Val/Test)
│   ├── train_category.py           # Fine-Tuning Skript (DistilBERT)
│   ├── train_sentiment.py          # Fine-Tuning Skript (RoBERTa)
│   └── evaluate.py                 # Evaluierung (Accuracy, Precision, Recall, F1)
│
├── data/                           # Datensätze (in .gitignore)
│   ├── raw/                        # Kaggle Originaldateien
│   └── processed/                  # Vorverarbeitete Datensätze
│
├── models/                         # Vortrainierte Basismodelle und feingetunte Gewichte
│   ├── base/
│   │   ├── distilbert-base-uncased/
│   │   └── roberta-base/
│   └── fine_tuned/
│       ├── category_model/
│       └── sentiment_model/
│
├── notebooks/                      # Jupyter Notebooks für Analysen
│   ├── 01_data_exploration.ipynb
│   └── 02_error_analysis.ipynb
│
├── tests/                          # Automated Testing Suite
│   ├── test_api.py                 # API-Endpunkt-Tests
│   ├── test_inference.py           # Inferenz-Pipeline Tests
│   └── test_priority.py           # Business-Logik Tests
│
├── docs/                           # Dokumentation
│   ├── architecture.md
│   ├── api.md
│   └── training.md
│
├── .env.example                    # Beispiel-Umgebungsvariablen
├── .gitignore
├── Dockerfile                      # Multistage CPU-optimiertes Dockerfile
├── requirements.txt                # Production Dependencies (Schlank)
├── requirements-dev.txt            # Development & Training Dependencies
├── README.md
├── LICENSE
└── pyproject.toml                  # Linter & Formatter Konfiguration (Black, Isort)
```

---

## 🚀 Quickstart (Lokale Einrichtung)

### 1. Repository klonen

```bash
git clone <repository-url>
cd customer-support-ticket-classifier
```

### 2. Virtuelle Umgebung aufsetzen

```bash
python -m venv .venv
source .venv/bin/activate  # Unter Windows: .venv\Scripts\activate
```

### 3. Abhängigkeiten installieren

Für das **lokale Training** und Entwickeln:

```bash
pip install --upgrade pip
pip install -r requirements-dev.txt
```

Für **nur die Inferenz / API** (Production Deployment):

```bash
pip install -r requirements.txt
```

### 4. Models herunterladen

```bash
python scripts/download_models.py
```

### 5. App starten
```bash
python -m uvicorn app.main:app --reload
```

---

## 🏋️‍♂️ Modell-Training (Phase 1)

1. **Daten vorbereiten:**
Lade den Kaggle-Datensatz herunter, platziere ihn unter `data/raw/` und starte die Vorbereitung:

```bash
python training/prepare_data.py
```


2. **Modelle feintunen:**

```bash
python training/train_category.py
python training/train_sentiment.py
```


Die fertigen Gewichte werden automatisch im performanten `.safetensors`-Format unter `models/fine_tuned/` abgelegt.
3. **Modelle evaluieren:**

```bash
python training/evaluate.py
```



---

## 🔌 API starten & testen (Phase 2)

### FastAPI Server lokal starten

```bash
uvicorn app.main:app --reload --port 8000
```

Die interaktive Swagger-Dokumentation ist anschließend unter **`http://localhost:8000/docs`** erreichbar.

### Beispiel-Anfrage (HTTP POST)

**Endpoint:** `POST /predict`

**Request Body:**

```json
{
  "text": "I cannot access my account after the latest system update and need urgent help!"
}
```

**Response Body:**

```json
{
  "category": "Login & Account Access",
  "category_confidence": 0.96,
  "sentiment": "Negative",
  "sentiment_confidence": 0.91,
  "priority": "High"
}
```

---

## 🧪 Tests & Code Quality

### Automated Unit- & Integrationstests

```bash
pytest --cov=app tests/
```

### Code Formatting & Linting

```bash
black app/ training/ tests/
isort app/ training/ tests/
flake8 app/ training/
```

---

## 🐳 Docker Deployment

Ein schlankes Docker-Image für Deployment-Plattformen (wie Hugging Face Spaces, Render oder AWS ECS) erstellen und ausführen:

```bash
# Docker Image bauen
docker build -t ticket-classifier .

# Container starten
docker run -p 8000:8000 ticket-classifier
```

---

## 📄 Lizenz

Dieses Projekt steht unter der MIT License