import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_fscore_support


def evaluate_outputs(outputs, include_source="causalquery"):
    pairs = [pair for result in outputs for pair in result["pairs"]]
    valid = [pair for pair in pairs if pair["dependence_score"] is not None and (include_source is None or pair["score_source"] == include_source)]
    if not valid:
        return {"total_pairs": len(pairs), "scored_pairs": 0, "coverage": 0.0, "auprc": None, "f1": None}
    targets = np.asarray([pair["true_functional_overlap"] for pair in valid], dtype=int)
    scores = np.asarray([pair["dependence_score"] for pair in valid], dtype=float)
    predictions = np.asarray([pair["predicted_functional_overlap"] for pair in valid], dtype=int)
    precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, average="binary", zero_division=0)
    ap = float(average_precision_score(targets, scores)) if np.any(targets == 1) else None
    return {
        "total_pairs": len(pairs),
        "scored_pairs": len(valid),
        "coverage": len(valid) / len(pairs),
        "functional_positive_pairs": int(np.sum(targets)),
        "structural_positive_pairs": int(sum(pair["true_structural_overlap"] for pair in valid)),
        "auprc": ap,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(np.sum((predictions == 1) & (targets == 1))),
        "fp": int(np.sum((predictions == 1) & (targets == 0))),
        "tn": int(np.sum((predictions == 0) & (targets == 0))),
        "fn": int(np.sum((predictions == 0) & (targets == 1))),
        "brier": float(brier_score_loss(targets, scores)),
    }
