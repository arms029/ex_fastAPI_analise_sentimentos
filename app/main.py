from fastapi import FastAPI

from app.api.routes import sentiment


def create_app() -> FastAPI:
    app = FastAPI(
        title="Sentiment Analysis API",
        description=(
            "Minimal viable product providing sentiment analysis using LangChain, "
            "LangGraph, and an Ollama-hosted language model."
        ),
        version="0.1.0",
    )

    app.include_router(sentiment.router, prefix="/sentiment", tags=["Sentiment"])

    return app


app = create_app()
