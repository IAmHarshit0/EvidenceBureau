import ollama
from sentence_transformers import SentenceTransformer, CrossEncoder
from evidence_bureau.retrieval import get_collection, query

MODEL_NAME = "all-MiniLM-L6-v2"
CHAT_MODEL = "qwen3.5:4b"
RETRIEVE_N = 10
RERANK_K = 5

collection = get_collection()
embed_model = SentenceTransformer(MODEL_NAME)
reranker = CrossEncoder("BAAI/bge-reranker-base")


def retrieve_context(question: str) -> str:
    result = query(collection, embed_model, question, n_results=RETRIEVE_N)
    ret_ids = result["ids"][0]
    ret_docs = result["documents"][0]

    pairs = [[question, doc] for doc in ret_docs]
    scores = reranker.predict(pairs)

    reranked = sorted(zip(ret_ids, ret_docs, scores), key=lambda x: x[2], reverse=True)
    top_chunks = reranked[:RERANK_K]

    context = "\n\n".join(
        f"[{chunk_id}]\n{doc}" for chunk_id, doc, _ in top_chunks
    )
    return context


def build_messages(question: str, context: str, history: list) -> list:
    system_prompt = (
        "You are a helpful assistant answering questions using ONLY the "
        "provided context below. If the answer isn't in the context, say "
        "you don't know — do not make anything up.\n\n"
        f"Context:\n{context}"
    )
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": question})
    return messages


def chat_loop():
    history = []
    print("RAG chat ready. Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        context = retrieve_context(question)
        messages = build_messages(question, context, history)

        response = ollama.chat(model=CHAT_MODEL, messages=messages)
        answer = response.message.content

        print(f"\nAssistant: {answer}\n")

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    chat_loop()