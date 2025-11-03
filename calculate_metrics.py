import json
from sklearn.metrics import precision_score, recall_score, f1_score

def load_labels(file_path):
    labels_list = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            labels_list.append(data["labels"])
    return labels_list

def compute_metrics(pred_file, gold_file):
    preds = load_labels(pred_file)
    golds = load_labels(gold_file)

    assert len(preds) == len(golds), "Số lượng sample không khớp!"

    em_count = 0
    all_preds = []
    all_golds = []

    for p_labels, g_labels in zip(preds, golds):
        # Exact Match
        if p_labels == g_labels:
            em_count += 1

        # Chuyển yes/no thành binary (yes=1, no=0) cho mỗi label
        all_preds.extend([1 if label == "yes" else 0 for label in p_labels])
        all_golds.extend([1 if label == "yes" else 0 for label in g_labels])

    em = em_count / len(preds)
    precision = precision_score(all_golds, all_preds, zero_division=0)
    recall = recall_score(all_golds, all_preds, zero_division=0)
    f1 = f1_score(all_golds, all_preds, zero_division=0)

    return {
        "Exact Match": em,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }

if __name__ == "__main__":
    pred_file = "results.txt"
    gold_file = "ground_truth_public_test.txt"

    metrics = compute_metrics(pred_file, gold_file)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

