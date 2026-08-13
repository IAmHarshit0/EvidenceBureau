from pathlib import Path

from src.embedding import load_chunks, create_embeddings

CHUNKED_FILE = Path("data/chunked_output.json")

def test_embeddings():
    chunks = load_chunks(CHUNKED_FILE)
    embeddings = create_embeddings(chunks)

    assert len(embeddings) == len(chunks)
    assert embeddings.shape[1] == 384