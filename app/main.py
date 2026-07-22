from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.inference import predict_ticket
from app.model_handler import model_handler
from app.schemas import ErrorResponse, HealthResponse, TicketRequest, TicketResponse

# Logging Konfiguration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI Lifespan Manager.

    Handles application startup and shutdown events.
    The parameter is prefixed with '_' to indicate it is required by FastAPI but unused.
    """
    logger.info("Starting up application: %s", settings.PROJECT_NAME)
    logger.info("Environment: %s | Device: %s", settings.API_ENV, settings.DEVICE)
    
    try:
        model_handler.load_models()
        logger.info("Application startup complete. Ready to serve requests.")
    except Exception:
        logger.exception("Application startup failed due to model loading error.")
        raise

    yield  # Application running state

    logger.info("Shutting down application...")
    model_handler.unload_models()
    logger.info("Application shutdown complete.")


# FastAPI Instanz erzeugen
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.APP_VERSION,
    description=(
        "Production-ready FastAPI service for classifying customer support tickets "
        "by category and sentiment with deterministic priority calculation."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Exception Handlers (Uniform JSON Error Response Format)
# -----------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handles Pydantic validation errors (e.g., empty string input, invalid schema)."""
    logger.warning("Validation error on request %s: %s", request.url.path, exc.errors())
    
    error_payload = ErrorResponse(
        status=status.HTTP_422_UNPROCESSABLE_ENTITY,
        error="Unprocessable Entity",
        message="Input validation failed.",
        path=request.url.path,
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_payload.model_dump(),
    )


@app.exception_handler(RuntimeError)
async def runtime_exception_handler(
    request: Request, exc: RuntimeError
) -> JSONResponse:
    """Handles runtime operational issues (e.g., models not initialized)."""
    logger.error("Runtime error on request %s: %s", request.url.path, exc)
    
    error_payload = ErrorResponse(
        status=status.HTTP_503_SERVICE_UNAVAILABLE,
        error="Service Unavailable",
        message=str(exc),
        path=request.url.path,
    )
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=error_payload.model_dump(),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(
    request: Request, exc: HTTPException
) -> JSONResponse:
    """Ensures HTTP exceptions adhere to the uniform JSON error response format."""
    logger.warning("HTTP Exception %d on %s: %s", exc.status_code, request.url.path, exc.detail)
    
    error_payload = ErrorResponse(
        status=exc.status_code,
        error="HTTP Exception",
        message=str(exc.detail),
        path=request.url.path,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=error_payload.model_dump(),
    )


# -----------------------------------------------------------------------------
# System Endpoints
# -----------------------------------------------------------------------------
@app.get(
    "/live",
    status_code=status.HTTP_200_OK,
    summary="Application Liveness Probe",
    tags=["System"],
)
def liveness_check() -> dict[str, str]:
    """Simple probe for orchestrators (Kubernetes/Render) to verify the process is alive."""
    return {"status": "alive"}


@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Application Readiness Probe",
    tags=["System"],
)
def health_check() -> HealthResponse:
    """Endpoint to check if the API and underlying ML models are initialized and ready."""
    is_ready = model_handler.is_healthy()

    if not is_ready:
        logger.warning("Readiness probe failed: Models are not ready.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models are not initialized or ready.",
        )

    return HealthResponse(
        status="healthy",
        environment=settings.API_ENV,
        version=settings.APP_VERSION,
        use_fine_tuned_models=settings.USE_FINE_TUNED_MODELS,
        models_loaded=is_ready,
        device=str(settings.DEVICE),
        category_model=settings.CATEGORY_MODEL_DIR.name,
        sentiment_model=settings.SENTIMENT_MODEL_DIR.name,
    )


# -----------------------------------------------------------------------------
# Inference Endpoints
# -----------------------------------------------------------------------------
@app.post(
    "/predict",
    response_model=TicketResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify Support Ticket",
    tags=["Inference"],
)
def predict(request: TicketRequest) -> TicketResponse:
    """Classifies an incoming customer support ticket.

    Synchronous endpoint ensuring CPU-bound PyTorch operations run smoothly in
    FastAPI's external threadpool without blocking the event loop.
    """
    return predict_ticket(request)