import chromadb
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
db_path = Path("data/db")
eval_path = Path("data/eval.json")

def get_collection():
    client = chromadb.PersistentClient(path=db_path)
    return client.get_or_create_collection(name="knowledge")

def query(collection, model, question: str, n_results: int = 5):
    query_embedding = model.encode([question])
    return collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results
    )

