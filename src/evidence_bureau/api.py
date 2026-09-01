# PYTHONPATH=src uv run uvicorn evidence_bureau.api:app --reload --port 8000

import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from evidence_bureau.slm import generate_answer, stream_answer, collection, CHAT_MODEL
import uuid
from pathlib import Path
from fastapi import UploadFile, File
from evidence_bureau.process_doc import process_document

app = FastAPI(title="Evidence Bureau API")

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_FILE_SIZE_MB = 25

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

@app.post("/process_document")
async def process_document_endpoint(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({size_mb:.1f}MB). Max {MAX_FILE_SIZE_MB}MB.",
        )

    source_id = uuid.uuid4().hex[:8]
    dest_path = UPLOAD_DIR / f"{source_id}_{file.filename}"
    with open(dest_path, "wb") as f:
        f.write(contents)

    try:
        result = process_document(dest_path, source_id=source_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")

    return {"status": "success", **result}