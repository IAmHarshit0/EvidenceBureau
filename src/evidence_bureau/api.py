# PYTHONPATH=src uv run uvicorn evidence_bureau.api:app --reload --port 8000

import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from evidence_bureau.slm import generate_answer, stream_answer, collection, CHAT_MODEL

app = FastAPI(title="Evidence Bureau API")

# Dev-friendly CORS. Tighten allow_origins to your actual frontend URL(s) before deploying.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    retrieve_n: int | None = Field(default=None, ge=1, le=50)
    rerank_k: int | None = Field(default=None, ge=1, le=20)
    stream: bool = Field(default=True)


@app.get("/health")
def health():
    try:
        count = collection.count()

        return {
            "status": "ok",
            "model": CHAT_MODEL,
            "collection_count": count,
        }

    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=str(e),
        )


@app.post("/ask")
def ask(payload: AskRequest):
    if not payload.stream:
        result = generate_answer(
            payload.question,
            retrieve_n=payload.retrieve_n,
            rerank_k=payload.rerank_k,
        )
        return result

    def event_generator():
        for event in stream_answer(
            payload.question,
            retrieve_n=payload.retrieve_n,
            rerank_k=payload.rerank_k,
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")