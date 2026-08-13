import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
db_path = Path("data/db")

client = chromadb.PersistentClient(path=db_path)
collection = client.get_or_create_collection(name="knowledge")
model = SentenceTransformer(MODEL_NAME)


def test_collection_not_empty():
    assert collection.count() > 0


def test_get_by_id():
    result = collection.get(limit=1)
    assert result["ids"]
    sample_id = result["ids"][0]
    fetched = collection.get(ids=[sample_id])
    assert fetched["ids"][0] == sample_id


def test_query():
    query_embedding = model.encode(["What happened in the personality experiment?"])
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=3
    )
    assert results["ids"][0]


if __name__ == "__main__":
    test_collection_not_empty()
    test_get_by_id()
    test_query()