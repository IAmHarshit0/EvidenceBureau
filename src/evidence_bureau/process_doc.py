import uuid
from pathlib import Path

from evidence_bureau.ingestion import extract_pdf
from evidence_bureau.chunking import chunk_pages
from evidence_bureau.embedding import create_embeddings
from evidence_bureau.resources import collection, embed_model


def process_document(pdf_path: Path, source_id: str | None = None) -> dict:
    """
    Extracts, chunks, embeds, and stores a PDF into the shared Chroma collection.
    source_id: a stable identifier prefixed onto every chunk_id to prevent
    collisions between documents. Defaults to a short random id if not given.
    """
    source_id = source_id or uuid.uuid4().hex[:8]

    extracted = extract_pdf(pdf_path)
    chunked = chunk_pages(extracted)

    chunks = [chunk for page in chunked["pages"] for chunk in page["chunks"]]

    if not chunks:
        return {
            "filename": chunked["filename"],
            "source_id": source_id,
            "pages": chunked["numberOfPages"],
            "chunks_stored": 0,
            "warning": "No extractable text found — the PDF may be scanned/image-based.",
        }

    embeddings = create_embeddings(chunks, embed_model)

    ids, documents, metadatas = [], [], []
    for chunk in chunks:
        ids.append(f"{source_id}__{chunk['chunk_id']}")
        documents.append(chunk["text"])
        metadatas.append({
            "source_id": source_id,
            "filename": chunked["filename"],
            "chunk_id": chunk["chunk_id"],
            "page_number": " ".join(chunk["chunk_id"].split("_")[:2]),
        })

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )

    return {
        "filename": chunked["filename"],
        "source_id": source_id,
        "pages": chunked["numberOfPages"],
        "chunks_stored": len(ids),
    }