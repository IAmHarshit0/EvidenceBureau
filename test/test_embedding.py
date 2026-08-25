from pathlib import Path
from sentence_transformers import SentenceTransformer
from src.evidence_bureau.embedding import load_chunks, create_embeddings

CHUNKED_FILE = Path("data/chunked_output.json")
MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)
def test_embeddings():
    chunks = load_chunks(CHUNKED_FILE)
    embeddings = create_embeddings(chunks, model)

    assert len(embeddings) == len(chunks)
    assert embeddings.shape[1] == 384