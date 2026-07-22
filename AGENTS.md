# AGENTS.md

## Project Overview

This repository contains a **Customer Support Ticket Classifier**, an NLP-based microservice that automatically classifies customer support tickets, detects their sentiment, and derives a business priority using deterministic business rules.

The project is divided into two phases:

1. **Training**
   - Prepare datasets.
   - Fine-tune Transformer models locally.
   - Evaluate model performance.
   - Save trained models.

2. **Inference**
   - Load trained models.
   - Expose a REST API using FastAPI.
   - Predict ticket category.
   - Predict ticket sentiment.
   - Calculate ticket priority.

---

## Tech Stack

- Python 3.10+
- FastAPI
- PyTorch (CPU)
- Hugging Face Transformers
- Datasets
- Scikit-Learn
- Pydantic v2+
- Docker
- Pytest

---

## Repository Structure

```text
app/
training/
tests/
data/
models/
docs/
notebooks/
```

---

## Coding Guidelines

### General

- Follow **PEP 8**.
- Use **type hints** whenever possible.
- Keep functions small and focused.
- Avoid duplicated code.
- Prefer composition over inheritance.
- Every public function should include a docstring.

### Tooling & Code Style

- Code must pass formatting with `black` and `isort`.
- Linting must pass with `flake8`.

---

## Project Architecture

Business logic must stay strictly separated.

### app/

Contains only inference logic.

**Allowed:**
- FastAPI endpoints
- Model loading & handler
- Prediction logic
- Priority calculation
- Request validation & schemas

**Not allowed:**
- Model training
- Dataset preprocessing
- Experimentation code

---

### training/

Contains only training logic.

**Allowed:**
- Data preprocessing
- Model fine-tuning
- Evaluation
- Metrics generation

**Not allowed:**
- FastAPI code
- HTTP endpoints

---

### tests/

Contains:
- Unit Tests
- Integration Tests

*Rule:* Every new feature or endpoint must include corresponding tests.

---

## Model Guidelines & Storage

- Models are always loaded locally from `models/`.
- Model weights must be saved in **SafeTensors** format (`model.safetensors`).
- **Never download models during inference.** Inference must work completely offline after local training.
- Model weights belong in `models/` and should **never** be committed to Git.

---

## Data Management

- Raw datasets belong in `data/raw/`.
- Processed datasets belong in `data/processed/`.
- Datasets should **never** be committed to Git.

---

### Separation of Concerns

Each module should have a single responsibility.

- `preprocessing.py` handles text normalization.
- `model_handler.py` is responsible for loading and managing models.
- `inference.py` performs model inference.
- `priority_engine.py` derives business priority.
- `main.py` contains only FastAPI routing and dependency wiring.

---

## Priority Engine

- Priority is **not predicted** by the ML models.
- Priority is calculated using deterministic business rules based on:
  - predicted category
  - predicted sentiment
- The priority engine must remain independent from the ML models.

---

## API & Error Handling

- The API should expose at least:
  - `GET /health`
  - `POST /predict`
- Future endpoints should remain RESTful.
- All incoming requests must be validated using **Pydantic schemas**.
- Automatically reject empty strings, invalid text formats, or missing fields with `HTTP 422 (Unprocessable Entity)`.
- Return proper HTTP status codes, never expose full raw stack traces to the client, and return meaningful error messages.

---

## Performance & Lifecycle

- Models **MUST** be initialized during application startup using FastAPI's `lifespan` context manager.
- **Never re-initialize or reload models inside request handlers.**
- Prefer:
  - Reusable model instances
  - Minimal memory usage
  - CPU-friendly inference
  - Lazy loading where appropriate

---

## Dependencies

- Production dependencies belong in `requirements.txt`.
- Training & development dependencies belong in `requirements-dev.txt`.
- Do not add unnecessary packages. Keep the production environment as lightweight as possible.

---

### Dependency Injection

Avoid global state whenever possible.

Use FastAPI dependency injection for services and shared resources.

---

## Docker Rules

Docker images should:
- Use CPU-only PyTorch (`--extra-index-url https://download.pytorch.org/whl/cpu`).
- Stay as small as possible.
- Avoid unnecessary build layers or dev-dependencies.

---

## Git Conventions

Use **Conventional Commits**.

Examples:
```text
feat: add sentiment inference
feat: implement priority engine
fix: correct category mapping
refactor: simplify model handler
test: add API integration tests
docs: update README
chore: update dependencies
```

---

## Documentation

Keep the following documents updated when making architectural or API changes:
- `README.md`
- `docs/api.md`
- `docs/architecture.md`

---

## Logging

Use Python's built-in logging module.

Do not use print() statements.

Log

- application startup
- model loading
- prediction requests
- handled exceptions

Do not log sensitive ticket content.

---

## Configuration

Application configuration must come from environment variables.

Do not hardcode

- file paths
- model locations
- ports
- secrets

Use `.env` only for local development.

---

## Type Safety

Use explicit type hints.

Avoid `Any` whenever possible.

Use Pydantic models for API contracts.

---

## Security

Never expose

- local file paths
- stack traces
- model internals

Always validate external input.

---

## Application Lifecycle

Models must be loaded exactly once during application startup using FastAPI's lifespan context.

Models must never be loaded inside request handlers.

---

## Future Extensions

Possible future improvements include:
- MLflow integration
- Model versioning
- Hugging Face Hub support
- ONNX optimization
- Multi-language classification
- Authentication & DB integration
- Monitoring & Kubernetes deployment

The architecture must remain modular so that these extensions can be added without major refactoring.
