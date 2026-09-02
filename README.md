# Evidence Bureau

An evidence-first local research assistant that lets you upload PDFs, retrieve relevant evidence, and ask questions using a local LLM.

Built to demonstrate **RAG, evaluation, telemetry, and production engineering** in one end-to-end application.

## Overview

```text
PDF
 │
 ▼
Extract → Chunk → Embed → ChromaDB
                         │
Question ────────────────┘
 │
 ▼
Retrieve → Rerank → Ollama
                  │
                  ▼
                Answer
```

The system retrieves and reranks relevant document chunks before generating an answer. The LLM is instructed to use only the provided evidence and to identify when the evidence is insufficient.

## Features

- PDF upload and text extraction
- Semantic retrieval with ChromaDB
- `all-MiniLM-L6-v2` embeddings
- `BAAI/bge-reranker-base` reranking
- Local LLM inference with Ollama (`qwen3.5:4b`)
- Streaming and non-streaming answers
- Request telemetry and timing
- Retrieval evaluation with Recall@5
- React frontend + FastAPI backend
- Dockerized application
- CI with GitHub Actions

## Tech Stack

| Component | Technology |
|---|---|
| Backend | FastAPI |
| Frontend | React |
| RAG | ChromaDB + Sentence Transformers |
| Reranker | BGE Reranker |
| LLM | Ollama + Qwen 3.5 4B |
| PDF | PyMuPDF |
| Testing | pytest |
| Package Manager | uv |
| Deployment | Docker + Docker Compose |
| CI | GitHub Actions |

## Running Locally

### Requirements

- Docker
- Docker Compose
- Ollama (installed and running on the host)

Ollama runs directly on the host machine, not inside Docker. Pull the model before starting the app:

```bash
ollama pull qwen3.5:4b
```

Start the application:

```bash
docker compose up -d --build
```

Then open:

```text
Frontend: http://localhost:5173
Backend:  http://localhost:8000
```

Stop the application:

```bash
docker compose down
```

### Configuration

The backend reads its configuration from environment variables (see `compose.yaml`). Defaults work out of the box. To override any of these, copy `.env.example` to `.env` at the project root and edit as needed:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|---|---|---|
| `CHAT_MODEL` | `qwen3.5:4b` | Ollama model used for generation |
| `EMBED_MODEL_NAME` | `all-MiniLM-L6-v2` | Sentence-transformer embedding model |
| `RERANKER_MODEL_NAME` | `BAAI/bge-reranker-base` | Cross-encoder reranker model |
| `HF_TOKEN` | — | Optional Hugging Face token for higher download rate limits |
| `BACKEND_PORT` | `8000` | Host port for the FastAPI backend |
| `FRONTEND_PORT` | `5173` | Host port for the React frontend |

## Testing

Run the test suite with:

```bash
uv run pytest
```

The tests cover the API, question answering, streaming, PDF processing, document upload, validation, and vector storage.

## Evaluation

The retrieval pipeline was evaluated on a project-specific dataset.

```text
Recall@5: 77.78%
```

Retrieval uses 15 candidates followed by cross-encoder reranking to the top 5 evidence chunks.

## CI

GitHub Actions runs on every push and pull request:

- spins up Ollama as a service container, pulls the chat model, and runs the test suite
- builds the backend and frontend Docker images to verify they build cleanly

## Architecture

```text
┌──────────────────┐
│    React UI      │
│   :5173          │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│    FastAPI       │
│   :8000          │
│    (Docker)      │
└───────┬──────────┘
        │
   ┌────┴─────┐
   ▼          ▼
 RAG       Ollama
 Pipeline   (host)
   │
   ▼
ChromaDB
```

The frontend and backend run in Docker; Ollama runs directly on the host and is reached via `host.docker.internal`.

## Project Structure

```text
evidence-bureau/
├── src/evidence_bureau/
│   ├── api.py
│   ├── chunking.py
│   ├── embedding.py
│   ├── ingestion.py
│   ├── process_doc.py
│   ├── resources.py
│   ├── retrieval.py
│   ├── slm.py
│   ├── telemetry.py
│   └── vector.py
├── frontend/
├── test/
├── data/
├── evals/
├── Dockerfile.backend
├── compose.yaml
├── .env.example
├── pyproject.toml
└── README.md
```

## Author

**Harshit Mishra**