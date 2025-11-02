from __future__ import annotations

from typing import TypedDict

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph


class SentimentState(TypedDict, total=False):
    """State tracked throughout the sentiment analysis graph."""

    input_text: str
    label: str
    confidence: float
    reasoning: str


class SentimentResult(BaseModel):
    """Structured output returned by the LLM."""

    label: str = Field(..., description="Sentiment label such as positive, negative or neutral.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the sentiment classification.")
    reasoning: str = Field(..., description="Short natural language explanation supporting the label.")


class SentimentGraphService:
    """Service that orchestrates sentiment analysis with LangChain and LangGraph."""

    def __init__(self, model: str = "llama3") -> None:
        self._llm = ChatOllama(model=model, temperature=0.0)
        self._parser = JsonOutputParser(pydantic_object=SentimentResult)
        self._chain = self._build_chain()
        self._graph = self._build_graph()

    def _build_chain(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert sentiment analyst. Classify the user's text as "
                    "positive, negative, or neutral. Respond in JSON matching the schema.",
                ),
                (
                    "human",
                    "Text: {text}\n\nReturn JSON with keys label, confidence, reasoning. "
                    "Confidence must be a float between 0 and 1.",
                ),
            ]
        )
        return prompt | self._llm | self._parser

    def _build_graph(self):
        graph = StateGraph(SentimentState)

        def classify(state: SentimentState) -> SentimentState:
            result = self._chain.invoke({"text": state["input_text"]})
            return {
                "input_text": state["input_text"],
                "label": result.label,
                "confidence": float(result.confidence),
                "reasoning": result.reasoning,
            }

        graph.add_node("classify", classify)
        graph.set_entry_point("classify")
        graph.add_edge("classify", END)
        return graph.compile()

    def analyse(self, text: str) -> SentimentResult:
        state: SentimentState = {"input_text": text}
        result_state = self._graph.invoke(state)
        return SentimentResult(**result_state)
