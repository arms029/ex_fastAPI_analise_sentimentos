from pydantic import BaseModel, Field


class SentimentRequest(BaseModel):
    text: str = Field(..., description="Input text to analyse for sentiment.")


class SentimentResponse(BaseModel):
    label: str = Field(..., description="Predicted sentiment label (e.g. positive, negative, neutral).")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1.")
    reasoning: str = Field(..., description="Model explanation for the prediction.")
