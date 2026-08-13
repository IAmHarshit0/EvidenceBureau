import chromadb
from pathlib import Path
from embedding import load_chunks, create_embeddings
from sentence_transformers import SentenceTransformer

CHUNKED_FILE = Path("data/chunked_output.json")
MODEL_NAME = "all-MiniLM-L6-v2"
db_path = Path("data/db")

chunks = load_chunks(CHUNKED_FILE)
embeddings = create_embeddings(chunks)
model = SentenceTransformer(MODEL_NAME)

client = chromadb.PersistentClient(path=db_path)
collection = client.get_or_create_collection(name="knowledge")

ids = []
documents = []
metadatas = []

for i, chunk in enumerate(chunks):
    ids.append(chunk["chunk_id"])
    documents.append(chunk["text"])
    metadatas.append({
        "page_number": " ".join(chunk["chunk_id"].split("_")[:2])
        })

collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings.tolist(),
    metadatas=metadatas
)

print(f"Stored {collection.count()} chunks in ChromaDB")

# result = collection.get(
#     ids=[ids[0]]
# )

# print(result)

query = "What happened in the personality experiment?"

query_embedding = model.encode([query])

results = collection.query(
    query_embeddings=query_embedding.tolist(),
    n_results=5
)

print(results)