from fastapi import APIRouter, Depends

from app.models.sentiment import SentimentRequest, SentimentResponse
from app.services.sentiment_graph import SentimentGraphService

router = APIRouter()


def get_service() -> SentimentGraphService:
    return SentimentGraphService()


@router.post("/analyse", response_model=SentimentResponse)
async def analyse_sentiment(
    payload: SentimentRequest,
    service: SentimentGraphService = Depends(get_service),
) -> SentimentResponse:
    result = service.analyse(payload.text)
    return SentimentResponse(**result.dict())
