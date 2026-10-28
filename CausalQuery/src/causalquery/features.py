import math


FEATURE_NAMES = ("covariance", "disagreement", "or_correction", "or_eligible", "availability")


def sigmoid(value):
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    result = math.exp(value)
    return result / (1.0 + result)


def paired_features(left, right, p, gamma, smoothing, total_probes, reference_probability=0.5):
    if len(left) != len(right) or total_probes < 1:
        raise ValueError("Mismatched signature lengths or invalid probe budget")
    paired = [(int(a["bit"]), int(b["bit"])) for a, b in zip(left, right) if a["status"] == "ok" and b["status"] == "ok" and a["bit"] in {0, 1} and b["bit"] in {0, 1}]
    count = len(paired)
    if count == 0:
        return {"count": 0, "features": None, "stats": None}
    qi = sum(a == 0 for a, b in paired) / count
    qj = sum(b == 0 for a, b in paired) / count
    qij = sum(a == 0 and b == 0 for a, b in paired) / count
    distance = sum(a != b for a, b in paired) / count
    covariance = qij - qi * qj
    x_cov = max(4.0 * covariance, 0.0)
    variance = min(4.0 * qi * (1.0 - qi), 4.0 * qj * (1.0 - qj))
    x_disagreement = max(1.0 - 2.0 * distance, 0.0) * variance
    eligible = int(qij > 0.0 and 0.0 < qi <= 0.5 and 0.0 < qj <= 0.5)
    or_correction = 0.0
    estimated_overlap = None
    estimated_sizes = None
    if eligible:
        log_u = math.log1p(-p)
        ai = math.log(qi) / log_u
        aj = math.log(qj) / log_u
        estimated_overlap = math.log(qi * qj / qij) / log_u
        gi = sigmoid(((1.0 - reference_probability) ** (ai - 1.0) - gamma) / smoothing)
        gj = sigmoid(((1.0 - reference_probability) ** (aj - 1.0) - gamma) / smoothing)
        or_correction = gi * gj * min(max(estimated_overlap, 0.0) / min(ai, aj), 1.0)
        estimated_sizes = [ai, aj]
    return {
        "count": count,
        "features": [x_cov, x_disagreement, or_correction, float(eligible), count / total_probes],
        "stats": {
            "q_left": qi,
            "q_right": qj,
            "q_joint": qij,
            "disagreement": distance,
            "covariance": covariance,
            "or_eligible": bool(eligible),
            "estimated_or_overlap": estimated_overlap,
            "estimated_or_sizes": estimated_sizes,
        },
    }


def covariance_confidence_epsilon(provider_count, count, alpha):
    if count <= 0:
        return None
    m = provider_count + provider_count * (provider_count - 1) // 2
    return math.sqrt(math.log(2 * m / alpha) / (2 * count))
