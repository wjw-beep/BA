import itertools

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score

from .features import FEATURE_NAMES, covariance_confidence_epsilon, paired_features, sigmoid
from .oracle import pair_labels
from .probe import run_claim_probes


def prepare_claims(dataset, config, include_labels=True):
    claims = dataset.get("claims", [])
    if not isinstance(claims, list) or not claims:
        raise ValueError("Dataset must contain at least one claim")
    seen = set()
    prepared = []
    for claim in claims:
        identifier = str(claim["id"])
        if identifier in seen:
            raise ValueError(f"Duplicate claim ID: {identifier}")
        seen.add(identifier)
        response_matrix = run_claim_probes(claim, config)
        labels = pair_labels(claim, config) if include_labels else {}
        pairs = []
        for left, right in itertools.combinations([str(item["id"]) for item in claim["providers"]], 2):
            truth = labels.get((left, right), {})
            pairs.append({
                "left": left,
                "right": right,
                "left_responses": response_matrix["responses"][left],
                "right_responses": response_matrix["responses"][right],
                "labels": truth,
            })
        prepared.append({"claim": claim, "probes": response_matrix, "pairs": pairs})
    return prepared


def features_for_pairs(prepared_claims, config, smoothing, splits=None):
    results = []
    for prepared in prepared_claims:
        claim = prepared["claim"]
        if splits is not None and claim.get("split") not in splits:
            continue
        for pair in prepared["pairs"]:
            result = paired_features(
                pair["left_responses"], pair["right_responses"],
                config["protocol"]["retention_probability"],
                config["inference"]["influence_cutoff"],
                smoothing, config["protocol"]["probes"],
                config["inference"]["influence_probability"],
            )
            results.append({
                "claim_id": str(claim["id"]),
                "left": pair["left"],
                "right": pair["right"],
                "label": pair["labels"]["functional"],
                "structural_label": pair["labels"]["structural"],
                **result,
            })
    return results


def train_vectors(rows):
    valid = [row for row in rows if row["features"] is not None]
    if not valid:
        raise ValueError("No valid paired probes for training or validation")
    return np.asarray([row["features"] for row in valid], dtype=float), np.asarray([row["label"] for row in valid], dtype=int)


class DependenceModel:
    def __init__(self, coefficients, intercept, smoothing, threshold, retention_probability, gamma, reference_probability, metadata=None):
        self.coefficients = [float(value) for value in coefficients]
        self.intercept = float(intercept)
        self.smoothing = float(smoothing)
        self.threshold = float(threshold)
        self.retention_probability = float(retention_probability)
        self.gamma = float(gamma)
        self.reference_probability = float(reference_probability)
        self.metadata = metadata or {}
        if len(self.coefficients) != len(FEATURE_NAMES):
            raise ValueError("Unexpected number of model coefficients")

    def predict(self, features):
        if features is None:
            return None
        if len(features) != len(self.coefficients):
            raise ValueError("Feature dimensionality mismatch")
        return sigmoid(self.intercept + sum(a * b for a, b in zip(self.coefficients, features)))

    def verify_config(self, config):
        if abs(self.retention_probability - float(config["protocol"]["retention_probability"])) > 1e-12:
            raise ValueError("Saved model and configuration have different retention probabilities")
        if abs(self.gamma - float(config["inference"]["influence_cutoff"])) > 1e-12:
            raise ValueError("Saved model and configuration have different influence cutoffs")
        if abs(self.reference_probability - float(config["inference"]["influence_probability"])) > 1e-12:
            raise ValueError("Saved model and configuration have different reference probabilities")

    def as_dict(self):
        return {
            "architecture": "causalquery-logistic-v1",
            "feature_names": list(FEATURE_NAMES),
            "coefficients": self.coefficients,
            "intercept": self.intercept,
            "smoothing": self.smoothing,
            "threshold": self.threshold,
            "retention_probability": self.retention_probability,
            "influence_cutoff": self.gamma,
            "reference_probability": self.reference_probability,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, record):
        if record.get("architecture") != "causalquery-logistic-v1" or record.get("feature_names") != list(FEATURE_NAMES):
            raise ValueError("Unsupported model architecture or features")
        return cls(record["coefficients"], record["intercept"], record["smoothing"], record["threshold"], record["retention_probability"], record["influence_cutoff"], record["reference_probability"], record.get("metadata"))


def fit_model(prepared_claims, config):
    inference = config["inference"]
    candidates = []
    for smoothing in inference["smoothing_grid"]:
        train_rows = features_for_pairs(prepared_claims, config, smoothing, {"train"})
        dev_rows = features_for_pairs(prepared_claims, config, smoothing, {"dev"})
        X_train, y_train = train_vectors(train_rows)
        if len(np.unique(y_train)) != 2:
            raise ValueError("Training split must contain both functional classes")
        estimator = LogisticRegression(
            C=float(inference["regularization_C"]),
            max_iter=int(inference["max_iter"]),
            random_state=int(config["protocol"]["master_seed"]),
        )
        estimator.fit(X_train, y_train)
        X_dev, y_dev = train_vectors(dev_rows)
        dev_predictions = estimator.predict_proba(X_dev)[:, 1]
        ap = float(average_precision_score(y_dev, dev_predictions)) if np.any(y_dev == 1) else 0.0
        candidates.append((ap, -float(smoothing), smoothing, estimator, X_train.shape[0], len(y_dev), dev_predictions, y_dev))
    best = max(candidates, key=lambda item: (item[0], item[1]))
    _, _, smoothing, estimator, train_size, dev_size, scores, labels = best
    thresholds = []
    for threshold in inference["threshold_grid"]:
        pred = scores >= float(threshold)
        f1 = float(f1_score(labels, pred, zero_division=0))
        thresholds.append((f1, -float(threshold), float(threshold)))
    f1, _, threshold = max(thresholds)
    model = DependenceModel(
        estimator.coef_[0].tolist(), estimator.intercept_[0], smoothing, threshold,
        config["protocol"]["retention_probability"], inference["influence_cutoff"], inference["influence_probability"],
        {"train_pairs": train_size, "development_pairs": dev_size, "development_ap": float(best[0]), "development_f1": f1},
    )
    report = {
        "chosen_smoothing": float(smoothing),
        "chosen_threshold": threshold,
        "development_auprc": float(best[0]),
        "development_f1": f1,
        "train_valid_pairs": train_size,
        "development_valid_pairs": dev_size,
        "candidate_smoothing": [{"smoothing": float(item[2]), "development_auprc": float(item[0])} for item in candidates],
    }
    return model, report


def scored_claim(prepared, config, model, include_ground_truth=False):
    model.verify_config(config)
    claim = prepared["claim"]
    providers = claim["providers"]
    num_providers = len(providers)
    fallback = claim.get("copy_aware_scores", {})
    results = []
    for pair in prepared["pairs"]:
        left, right = pair["left"], pair["right"]
        feature = paired_features(
            pair["left_responses"], pair["right_responses"],
            config["protocol"]["retention_probability"],
            config["inference"]["influence_cutoff"],
            model.smoothing, config["protocol"]["probes"],
            config["inference"]["influence_probability"],
        )
        count = feature["count"]
        pair_key = "|".join(sorted((left, right)))
        if feature["features"] is not None:
            value = model.predict(feature["features"])
            source = "causalquery"
        elif pair_key in fallback:
            value = float(fallback[pair_key])
            if not 0 <= value <= 1:
                raise ValueError("Fallback dependence score must be inside [0,1]")
            source = "copy_aware_fallback"
        else:
            value = None
            source = "unresolved"
        epsilon = covariance_confidence_epsilon(num_providers, count, config["inference"]["confidence_alpha"])
        certificate = None if epsilon is None or (count < config["protocol"]["probes"] and not config["inference"].get("assume_noninformative_missingness", False)) else bool(feature["stats"]["covariance"] > 3.0 * epsilon)
        result = {
            "left": left,
            "right": right,
            "valid_paired_probes": count,
            "dependence_score": value,
            "predicted_functional_overlap": None if value is None else bool(value >= model.threshold),
            "score_source": source,
            "features": feature["features"],
            "statistics": feature["stats"],
            "covariance_confidence_epsilon": epsilon,
            "positive_or_overlap_certificate": certificate,
        }
        if include_ground_truth:
            result["true_functional_overlap"] = pair["labels"]["functional"]
            result["true_structural_overlap"] = pair["labels"]["structural"]
        results.append(result)
    from .ranking import greedy_rank
    ranked_scores = {tuple(sorted((item["left"], item["right"]))): item["dependence_score"] for item in results if item["dependence_score"] is not None}
    ranking = greedy_rank(
        providers, ranked_scores, config["ranking"]["lambda_penalty"],
        config["ranking"]["evidence_budget"], config["ranking"]["unknown_pair_penalty"],
    )
    return {
        "claim_id": str(claim["id"]),
        "split": claim.get("split"),
        "snapshot": str(claim["snapshot"]),
        "provider_count": num_providers,
        "probe_budget": config["protocol"]["probes"],
        "total_provider_calls": num_providers * config["protocol"]["probes"],
        "ranking": ranking,
        "pairs": results,
    }
