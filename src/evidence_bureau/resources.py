from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

DB_PATH = Path("data/db")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHAT_MODEL = "qwen3.5:4b"

_client = chromadb.PersistentClient(path=DB_PATH)
collection = _client.get_or_create_collection(name="knowledge")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
reranker = CrossEncoder("BAAI/bge-reranker-base")