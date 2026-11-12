"""
Softobotics Office Assistant RAG API.

Endpoints:

GET  /
GET  /health
POST /api/chat
"""

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import ChatRequest
from schemas import ChatResponse

from rag import ask_rag


app = FastAPI(
    title="Softobotics Office Assistant API",
    description=(
        "RAG chatbot API using LangChain, "
        "ChromaDB, Hugging Face embeddings, "
        "and Ollama."
    ),
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "Softobotics Office Assistant API"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


@app.post(
    "/api/chat",
    response_model=ChatResponse,
)
def chat(request: ChatRequest):
    """
    RAG chatbot endpoint.

    Flow:

    Question
        ↓
    Retriever
        ↓
    ChromaDB
        ↓
    Relevant Documents
        ↓
    Prompt
        ↓
    Ollama LLM
        ↓
    Answer
    """

    try:
        result = ask_rag(
            question=request.question
        )

        return ChatResponse(
            answer=result["answer"],
            sources=result.get(
                "sources",
                [],
            ),
        )

    except Exception as error:

        print(
            "RAG ERROR:",
            repr(error),
        )

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )