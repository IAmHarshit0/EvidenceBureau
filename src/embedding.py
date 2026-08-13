from pathlib import Path
import json

from sentence_transformers import SentenceTransformer


CHUNKED_FILE = Path("data/chunked_output.json")
MODEL_NAME = "all-MiniLM-L6-v2"


def load_chunks(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    chunks = []

    for page in data["pages"]:
        for chunk in page["chunks"]:
            chunks.append(chunk)

    return chunks


def create_embeddings(chunks: list[dict]):
    model = SentenceTransformer(MODEL_NAME)

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        show_progress_bar=True
    )

    return embeddings


if __name__ == "__main__":
    chunks = load_chunks(CHUNKED_FILE)

    embeddings = create_embeddings(chunks)

    print(f"Number of chunks: {len(chunks)}")
    print(f"Embedding shape: {embeddings.shape}")