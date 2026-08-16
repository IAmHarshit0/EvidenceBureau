import json
from evidence_bureau.retrieval import get_collection, query
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder

eval_path = Path("data/eval.json")
collection = get_collection()
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)
reranker = CrossEncoder("BAAI/bge-reranker-base")

RETRIEVE_N = 10   # cast a wide net with the bi-encoder
RERANK_K = 5      # then narrow down with the cross-encoder

with open(eval_path, "r") as e:
    eval_data = json.load(e)

total_recall = 0.0
eval_summary = []

for item in eval_data:
    question = item["question"]
    chunk_ids = item["relevant_ids"]

    # Stage 1: fast bi-encoder retrieval, wide net (top 10)
    result = query(collection, model, question, n_results=RETRIEVE_N)
    ret_ids = result["ids"][0]
    ret_docs = result["documents"][0]

    # Stage 2: cross-encoder reranks those 10 by relevance to the question
    pairs = [[question, doc] for doc in ret_docs]
    scores = reranker.predict(pairs)

    reranked = sorted(zip(ret_ids, ret_docs, scores), key=lambda x: x[2], reverse=True)
    reranked_ids = [chunk_id for chunk_id, _, _ in reranked][:RERANK_K]

    hits = set(chunk_ids).intersection(set(reranked_ids))
    recall_at_k = len(hits) / len(set(chunk_ids)) if set(chunk_ids) else 0.0

    total_recall += recall_at_k

    eval_summary.append(
        {
            "question": question,
            "ground_truth": chunk_ids,
            "retrieved_stage1": ret_ids,
            "reranked_top_k": reranked_ids,
            "hits": list(hits),
            "recall": round(recall_at_k, 4)
        }
    )

print(f"{(total_recall/len(eval_summary))*100:.2f}%")
# print(eval_summary)