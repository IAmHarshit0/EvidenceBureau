import os
from pathlib import Path
import chromadb
import ollama
from sentence_transformers import SentenceTransformer, CrossEncoder

DB_PATH = Path(os.getenv("CHROMA_DB_PATH", "data/db"))
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-base")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3.5:4b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

_client = chromadb.PersistentClient(path=str(DB_PATH))
collection = _client.get_or_create_collection(name="knowledge")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
reranker = CrossEncoder(RERANKER_MODEL_NAME)

ollama_client = ollama.Client(host=OLLAMA_HOST)