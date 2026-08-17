import json
import ollama
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer, CrossEncoder
from evidence_bureau.retrieval import get_collection, query

EVAL_PATH = Path("data/eval.json")
OUTPUT_PATH = Path("data/qa_output.json")

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHAT_MODEL = "qwen3.5:4b"
RETRIEVE_N = 15
RERANK_K = 5

collection = get_collection()
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
reranker = CrossEncoder("BAAI/bge-reranker-base")


def retrieve_context(question: str) -> str:
    result = query(collection, embed_model, question, n_results=RETRIEVE_N)
    ret_ids = result["ids"][0]
    ret_docs = result["documents"][0]

    pairs = [[question, doc] for doc in ret_docs]
    scores = reranker.predict(pairs)

    reranked = sorted(zip(ret_ids, ret_docs, scores), key=lambda x: x[2], reverse=True)
    top_chunks = reranked[:RERANK_K]

    return "\n\n".join(f"[{chunk_id}]\n{doc}" for chunk_id, doc, _ in top_chunks)


def ask(question: str, stream: bool = False):
    context = retrieve_context(question)
    system_prompt = (
        "You are Evidence Bureau, an evidence-first research analyst. "
        "Your job is to investigate questions using ONLY the provided evidence.\n\n"
        "RULES:\n"
        "1. Use only information contained in the provided context.\n"
        "2. Do not use your general knowledge or invent missing information.\n"
        "3. If the evidence is insufficient to answer the question, clearly say "
        "that the available evidence is insufficient.\n"
        "4. Distinguish between facts directly supported by the evidence and "
        "reasonable conclusions drawn from it.\n"
        "5. When possible, cite the page number or source identifier supporting "
        "your answer.\n"
        "6. Do not claim that the evidence says something when it does not.\n"
        "7. Answer directly and concisely. Do not narrate your reasoning process or restate these instructions. \n"
        "8. If the question contains a false assumption, point it out rather "
        "than accepting the assumption.\n\n"
        "STYLE:\n"
        "- Be concise and analytical.\n"
        "- Present the answer first, then supporting evidence.\n"
        "- Use bullet points when they improve clarity.\n"
        "- Do not unnecessarily repeat the context.\n"
        "- Maintain an investigative, neutral tone.\n\n"
        f"PROVIDED EVIDENCE:\n{context}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    if not stream:
        response = ollama.chat(model=CHAT_MODEL, messages=messages, options={"repeat_penalty": 1.3})
        return response.message.content

    # streaming path: yield content chunks as they arrive
    full_answer = ""
    for chunk in ollama.chat(model=CHAT_MODEL, messages=messages, options={"repeat_penalty": 1.3}, stream=True):
        piece = chunk["message"]["content"]
        print(piece, end="", flush=True)
        full_answer += piece
    print()  # newline after the streamed answer finishes
    return full_answer


def chat_loop():
    print("RAG chat ready. Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        print("\nAssistant: ", end="", flush=True)
        ask(question, stream=True)
        print()

def main():
    with open(EVAL_PATH, "r") as f:
        eval_data = json.load(f)

    results = []

    for i, item in enumerate(eval_data, 1):
        question = item["question"]
        print(f"[{i}/{len(eval_data)}] {question}")

        answer = ask(question)
        results.append({
            "question": question,
            "answer": answer
        })

    output = {
        "model": CHAT_MODEL,
        "generated_at": datetime.now().isoformat(),
        "results": results
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved {len(results)} Q&A pairs to {OUTPUT_PATH}")


if __name__ == "__main__":
    chat_loop()