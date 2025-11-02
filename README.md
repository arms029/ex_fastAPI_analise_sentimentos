# Sentiment Analysis FastAPI MVP

Mínimo produto viável de uma API de análise de sentimentos construída com [FastAPI](https://fastapi.tiangolo.com/), [LangChain](https://python.langchain.com/), [LangGraph](https://langchain-ai.github.io/langgraph/) e o modelo gratuito disponibilizado pelo [Ollama](https://ollama.com/).

## Arquitetura

```
app/
├── api/
│   └── routes/
│       └── sentiment.py        # Rotas REST para inferência
├── core/                       # Espaço reservado para configurações futuras
├── models/
│   └── sentiment.py            # Schemas Pydantic de request/response
└── services/
    └── sentiment_graph.py      # Orquestração LangChain + LangGraph
```

A lógica de inferência é encapsulada em um grafo de estados (`SentimentGraphService`) que combina:

1. Um `ChatPromptTemplate` do LangChain responsável por estruturar a instrução do modelo;
2. O conector `ChatOllama`, que consulta localmente um modelo gratuito (ex.: `llama3`);
3. Um `JsonOutputParser` para garantir uma resposta estruturada com `label`, `confidence` e `reasoning`;
4. Um `StateGraph` do LangGraph que mantém o fluxo simples (entrada → classificação → fim).

## Requisitos

- Python 3.11+
- [Ollama](https://ollama.com/download) instalado e executando localmente
- Modelo suportado pelo Ollama (por padrão, `ollama pull llama3`)

## Configuração do ambiente

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Executando a API

1. Garanta que o servidor do Ollama está ativo (`ollama serve`).
2. Inicie a aplicação FastAPI com Uvicorn:

```bash
uvicorn app.main:app --reload
```

A documentação interativa estará disponível em `http://127.0.0.1:8000/docs`.

## Exemplo de requisição

```bash
curl -X POST "http://127.0.0.1:8000/sentiment/analyse" \
  -H "Content-Type: application/json" \
  -d '{"text": "Eu adorei o novo produto!"}'
```

Resposta esperada (valores ilustrativos):

```json
{
  "label": "positive",
  "confidence": 0.92,
  "reasoning": "O texto expressa apreciação explícita pelo produto."
}
```

## Próximos passos sugeridos

- Cachear instâncias do grafo/LLM para reduzir latência por requisição;
- Persistir histórico das análises para auditoria;
- Expor métricas e health-checks.
