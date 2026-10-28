

def greedy_rank(providers, pair_scores, penalty, budget, unknown_pair_penalty):
    records = {str(item["id"]): item for item in providers}
    for item in providers:
        relevance = float(item["relevance"])
        if not 0 <= relevance <= 1:
            raise ValueError("Relevance scores must already be normalized to [0,1]")
    selected = []
    while len(selected) < min(budget, len(records)):
        candidates = []
        for identifier, item in records.items():
            if identifier in selected:
                continue
            scores = []
            for previous in selected:
                key = tuple(sorted((identifier, previous)))
                scores.append(pair_scores.get(key, unknown_pair_penalty))
            redundancy = max(scores, default=0.0)
            score = float(item["relevance"]) - penalty * redundancy
            candidates.append((score, float(item["relevance"]), identifier, redundancy))
        objective, relevance, chosen, redundancy = max(candidates, key=lambda x: (x[0], x[1], x[2]))
        selected.append(chosen)
    return selected
