import itertools


def export_public_responses(prepared_claims):
    claims = []
    for prepared in prepared_claims:
        claim = prepared["claim"]
        claims.append({
            "id": str(claim["id"]),
            "split": claim.get("split"),
            "snapshot": str(claim["snapshot"]),
            "providers": [{"id": str(provider["id"]), "relevance": float(provider["relevance"])} for provider in claim["providers"]],
            "probes": prepared["probes"],
            "copy_aware_scores": claim.get("copy_aware_scores", {}),
        })
    return {"format": "causalquery-public-responses-v1", "claims": claims}


def prepare_external_claims(data, config, include_labels=False, label_splits=None):
    if data.get("format") != "causalquery-public-responses-v1":
        raise ValueError("Unsupported external response format")
    prepared = []
    ids = set()
    for claim in data["claims"]:
        claim_id = str(claim["id"])
        if claim_id in ids:
            raise ValueError("Duplicate claim identifier")
        ids.add(claim_id)
        records = claim["providers"]
        provider_ids = [str(item["id"]) for item in records]
        if len(provider_ids) != len(set(provider_ids)):
            raise ValueError("Duplicate provider identifier")
        probes = claim["probes"]
        K = config["protocol"]["probes"]
        if len(probes["seeds"]) != K or len(set(probes["seeds"])) != K:
            raise ValueError("Invalid number of probes or repeated intervention seeds")
        if set(probes["responses"]) != set(provider_ids):
            raise ValueError("Response matrix provider identifiers do not match")
        if probes["snapshot"] != claim["snapshot"] or probes["claim_id"] != claim_id:
            raise ValueError("Snapshot or claim identifier mismatch")
        matrix = {}
        for provider in provider_ids:
            signature = probes["responses"][provider]
            if len(signature) != K:
                raise ValueError("Incomplete provider response signature")
            normalized = []
            for response in signature:
                if response["status"] == "ok" and response["snapshot"] == claim["snapshot"] and type(response["bit"]) is int and response["bit"] in {0, 1}:
                    normalized.append(response)
                else:
                    normalized.append({"status": "unavailable", "bit": None, "snapshot": response.get("snapshot")})
            matrix[provider] = normalized
        pairs = []
        need_labels = include_labels and (label_splits is None or claim.get("split") in label_splits)
        annotations = claim.get("pair_labels", {}) if need_labels else {}
        for left, right in itertools.combinations(provider_ids, 2):
            key = "|".join(sorted((left, right)))
            labels = annotations.get(key, {})
            if need_labels and "functional" not in labels:
                raise ValueError(f"Missing functional annotation for {claim_id}/{key}")
            pairs.append({"left": left, "right": right, "left_responses": matrix[left], "right_responses": matrix[right], "labels": labels})
        prepared.append({"claim": claim, "probes": probes, "pairs": pairs})
    return prepared
