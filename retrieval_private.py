from typing import List, Dict, Tuple, Any
from utils import load_yaml, read_jsonl
import json
import numpy as np
from sentence_transformers.util import cos_sim
from tqdm import tqdm

cfg = load_yaml("agent.yaml")
ret_cfg = cfg["retrieval"]
paths = cfg["paths"]

def _load_duration_train(path: str) -> List[Dict[str, Any]]:
    data = read_jsonl(path)
    print(f"Loaded duration train cases: {len(data)}")
    return data

def _case_text(sample: Dict[str, Any]) -> str:
    ctx = sample.get("context", "")
    q = sample.get("question", "")
    return f"{ctx.strip()} [SEP] {q.strip()}"
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('intfloat/multilingual-e5-large', cache_folder="cache_dir")
def _build_duration_embeddings(cases: List[Dict[str, Any]]) -> List[Tuple[List[float], int]]:
    texts = []
    for s in cases:
        texts.append("passage: " + _case_text(s))
    embs = model.encode(texts, normalize_embeddings=True)
    return embs

duration_cases = _load_duration_train(paths["duration_train_file"])
print('done _load_duration_train')
duration_index = _build_duration_embeddings(duration_cases)
print('done _build_duration_embeddings')

duration_cases_test = _load_duration_train(paths["duration_test_file_private"])
print('done _load_duration_train test')
duration_index_test = _build_duration_embeddings(duration_cases_test)
print('done _build_duration_embeddings test')
# 
similarities = cos_sim(duration_index_test, duration_index)
similarities = similarities.cpu().numpy()
# 
with open('similarities_private.npy', 'wb') as f:
    np.save(f, similarities)

similarities = np.load('similarities_private.npy')

def _retrieve_cases(i) -> List[Tuple[float, Dict[str, Any]]]:
    sims: List[Tuple[float, Dict[str, Any]]] = []

    similarities_i = similarities[i]
    for idx, s in enumerate(similarities_i):
        sims.append((s, duration_cases[idx]))
    sims.sort(key=lambda x: x[0], reverse=True)
    # filter by threshold (with fallback)
    keep = [(s, c) for (s, c) in sims[: ret_cfg["top_k"]] if s >= ret_cfg["min_sim"]]
    if not keep:
        # fallback lower
        keep = [(s, c) for (s, c) in sims[: ret_cfg["top_k"]] if s >= 0.25]
    return keep


def _solve_single_duration( context: str, question: str, options: List[str], i) -> List[str]:
    neighbors_scored  = _retrieve_cases(i)
    top_m = [c for _, c in neighbors_scored[: ret_cfg["top_m"]]]
    print(top_m)
    print('done retrieve')
    return top_m

if __name__ == "__main__":
    test_path = paths["duration_test_file_private"]
    results_retrieval = []
    for i, item in enumerate(tqdm(duration_cases_test)):
        context = item.get("context", "")
        question = item.get("question", "")
        options = item.get("options", [])
        qid = item.get("qid", None)

        labels = _solve_single_duration(context, question, options, i)
        results_retrieval.append(labels)

    with open("results_retrieval_private.json", 'w') as f:
        json.dump(results_retrieval, f, indent=4, ensure_ascii=False)
