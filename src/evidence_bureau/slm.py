import json
import ollama
from pathlib import Path
from datetime import datetime
from evidence_bureau.resources import collection, embed_model, reranker, CHAT_MODEL
from evidence_bureau.telemetry import (
    start_trace, finish_trace, save_trace, start_timer, elapsed_ms, record_error
)

EVAL_PATH = Path("data/eval.json")
OUTPUT_PATH = Path("data/qa_output.json")

DEFAULT_RETRIEVE_N = 15
DEFAULT_RERANK_K = 5


def build_system_prompt(context: str) -> str:
    return (
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


def retrieve_context(question: str, retrieve_n: int = None, rerank_k: int = None) -> tuple[str, dict]:
    n = retrieve_n or DEFAULT_RETRIEVE_N
    k = rerank_k or DEFAULT_RERANK_K

    timer = start_timer()

    result = collection.query(
        query_embeddings=embed_model.encode([question]).tolist(),
        n_results=n,
    )
    ret_ids = result["ids"][0]
    ret_docs = result["documents"][0]

    rerank_timer = start_timer()
    pairs = [[question, doc] for doc in ret_docs]
    scores = reranker.predict(pairs)
    rerank_ms = elapsed_ms(rerank_timer)

    reranked = sorted(zip(ret_ids, ret_docs, scores), key=lambda x: x[2], reverse=True)
    top_chunks = reranked[:k]

    context = "\n\n".join(f"[{chunk_id}]\n{doc}" for chunk_id, doc, _ in top_chunks)

    retrieval_meta = {
        "retrieve_n": n,
        "rerank_k": k,
        "retrieved_ids": ret_ids,
        "reranked_ids": [chunk_id for chunk_id, _, _ in top_chunks],
        "rerank_ms": rerank_ms,
        "retrieval_ms": elapsed_ms(timer),
    }

    return context, retrieval_meta


def generate_answer(question: str, retrieve_n: int = None, rerank_k: int = None) -> dict:
    """Non-streaming: retrieves context, calls the model once, returns full answer + metadata."""
    trace = start_trace()
    trace["question"] = question
    trace["model"] = CHAT_MODEL

    try:
        context, retrieval_meta = retrieve_context(question, retrieve_n, rerank_k)
        trace["retrieval"] = retrieval_meta
    except Exception as e:
        record_error(trace, e)
        save_trace(finish_trace(trace))
        raise

    messages = [
        {"role": "system", "content": build_system_prompt(context)},
        {"role": "user", "content": question},
    ]

    generation_timer = start_timer()
    try:
        response = ollama.chat(model=CHAT_MODEL, messages=messages, options={"repeat_penalty": 1.3})
        answer = response.message.content
    except Exception as e:
        trace["generation_ms"] = elapsed_ms(generation_timer)
        record_error(trace, e)
        save_trace(finish_trace(trace))
        raise

    trace["generation_ms"] = elapsed_ms(generation_timer)
    trace["answer"] = answer
    save_trace(finish_trace(trace))

    return {
        "answer": answer,
        "trace_id": trace["trace_id"],
        "retrieval": retrieval_meta,
        "generation_ms": trace["generation_ms"],
    }


def stream_answer(question: str, retrieve_n: int = None, rerank_k: int = None):
    """
    Streaming: yields dicts as generation progresses.
    Caller (CLI or API) decides how to render each piece.
    Always yields a final {"event": "done", ...} item, or {"event": "error", ...} on failure.
    """
    trace = start_trace()
    trace["question"] = question
    trace["model"] = CHAT_MODEL

    try:
        context, retrieval_meta = retrieve_context(question, retrieve_n, rerank_k)
        trace["retrieval"] = retrieval_meta
    except Exception as e:
        record_error(trace, e)
        save_trace(finish_trace(trace))
        yield {"event": "error", "message": str(e)}
        return

    yield {"event": "start", "trace_id": trace["trace_id"], "retrieval": retrieval_meta}

    messages = [
        {"role": "system", "content": build_system_prompt(context)},
        {"role": "user", "content": question},
    ]

    generation_timer = start_timer()
    full_answer = ""

    try:
        for chunk in ollama.chat(model=CHAT_MODEL, messages=messages, options={"repeat_penalty": 1.3}, stream=True):
            piece = chunk["message"]["content"]
            full_answer += piece
            yield {"event": "token", "content": piece}
    except Exception as e:
        trace["generation_ms"] = elapsed_ms(generation_timer)
        record_error(trace, e)
        save_trace(finish_trace(trace))
        yield {"event": "error", "message": str(e)}
        return

    trace["generation_ms"] = elapsed_ms(generation_timer)
    trace["answer"] = full_answer
    save_trace(finish_trace(trace))

    yield {"event": "done", "answer": full_answer, "generation_ms": trace["generation_ms"]}


def chat_loop():
    print("RAG chat ready. Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        print("\nAssistant: ", end="", flush=True)
        for event in stream_answer(question):
            if event["event"] == "token":
                print(event["content"], end="", flush=True)
            elif event["event"] == "error":
                print(f"\n[error] {event['message']}")
        print()


def main():
    with open(EVAL_PATH, "r") as f:
        eval_data = json.load(f)

    results = []

    for i, item in enumerate(eval_data, 1):
        question = item["question"]
        print(f"[{i}/{len(eval_data)}] {question}")

        result = generate_answer(question)
        results.append({
            "question": question,
            "answer": result["answer"]
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