from src.evidence_bureau.retrieval import get_collection, query
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

def test_collection_exists():
    collection = get_collection()
    assert collection.count() > 0

def test_query_results():
    results = query(collection=get_collection(), model=model, question="What is deception?")
    assert len(results["ids"][0]) == 5