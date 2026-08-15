import json
from retrieval import get_collection, query
from pathlib import Path
from sentence_transformers import SentenceTransformer

eval_path = Path("data/eval.json")
colletion = get_collection()
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

with open(eval_path, "r") as e:
    eval = json.load(e)

total_recall = 0.0
eval_summary = []

for item in eval:
    question = item["question"]
    chunk_ids = item["relevant_ids"]
    result = query(colletion, model, question)
    ret_ids = result["ids"][0]
    hits = set(chunk_ids).intersection(set(ret_ids))
    reacall_at_k = len(hits)/len(set(chunk_ids)) if set(chunk_ids) else 0.0

    total_recall += reacall_at_k

    eval_summary.append(
        {
            "question": question,
            "ground_truth" : chunk_ids,
            "retreived" : ret_ids,
            "hits" : list(hits),
            "recall" : round(reacall_at_k, 4)
        }
    )

print(f"{(total_recall/len(eval_summary))*100:.2f}%")
# print(eval_summary)