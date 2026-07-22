from enum import Enum
from pydantic import BaseModel, Field, field_validator


class PriorityEnum(str, Enum):
    """Enumeration of allowed ticket priority levels."""
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class TicketRequest(BaseModel):
    """Request schema for incoming customer support tickets."""
    
    text: str = Field(
        ...,
        description="The raw text content of the customer support ticket.",
        min_length=1,
        max_length=5000,
        examples=["I cannot log into my account since yesterday and getting an invalid token error."]
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, value: str) -> str:
        """Validates that the input text is not empty or composed solely of whitespace."""
        clean_value = value.strip()
        if not clean_value:
            raise ValueError("Ticket text cannot be empty or contain only whitespace.")
        return clean_value


class TicketResponse(BaseModel):
    """Response schema returned after ticket classification and priority assessment."""
    
    category: str = Field(
        ...,
        description="The predicted ticket category.",
        examples=["Login & Account Access"]
    )
    category_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence score for the predicted category (0.0 to 1.0).",
        examples=[0.96]
    )
    sentiment: str = Field(
        ...,
        description="The predicted sentiment score (e.g., Negative, Neutral, Positive, Angry).",
        examples=["Negative"]
    )
    sentiment_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence score for the predicted sentiment (0.0 to 1.0).",
        examples=[0.91]
    )
    priority: PriorityEnum = Field(
        ...,
        description="The calculated business priority based on deterministic rules.",
        examples=[PriorityEnum.HIGH]
    )

class HealthResponse(BaseModel):
    """Response schema for system health and readiness probes."""
    status: str = Field(..., examples=["healthy"])
    environment: str = Field(..., examples=["development"])
    version: str = Field(..., examples=["1.0.0"])
    use_fine_tuned_models: bool = Field(..., examples=[False])
    models_loaded: bool = Field(..., examples=[True])
    device: str = Field(..., examples=["cpu"])
    category_model: str = Field(..., examples=["distilbert-base-uncased"])
    sentiment_model: str = Field(..., examples=["roberta-base"])


class ErrorResponse(BaseModel):
    """Uniform error response schema across all exception handlers."""
    status: int = Field(..., examples=[503])
    error: str = Field(..., examples=["Service Unavailable"])
    message: str = Field(..., examples=["Models are not initialized or ready."])
    path: str = Field(..., examples=["/predict"])