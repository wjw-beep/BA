import itertools

import numpy as np

from .hashing import canonical_origin, seed_for_probe
from .provider import evaluate_rule, rule_origins


def origin_influences(record, config, claim_id="", provider_id=""):
    origins = sorted(canonical_origin(x) for x in record["referenced_origins"])
    rule = record["support"]
    active = rule_origins(rule)
    p = float(config["inference"]["influence_probability"])
    oracle = config["oracle"]
    rng_seed = seed_for_probe(config["protocol"]["master_seed"], claim_id, len(origins), str(provider_id))
    rng = np.random.default_rng(int(rng_seed[:16], 16))
    influences = {}
    for origin in origins:
        if origin not in active:
            influences[origin] = 0.0
            continue
        other = [item for item in origins if item != origin and item in active]
        if len(other) <= int(oracle["max_exact_origins"]):
            influence = 0.0
            for bits in itertools.product((False, True), repeat=len(other)):
                available = {item for item, bit in zip(other, bits) if bit}
                probability = p ** sum(bits) * (1.0 - p) ** (len(bits) - sum(bits))
                influence += probability * (evaluate_rule(rule, available | {origin}) != evaluate_rule(rule, available))
        else:
            samples = int(oracle["monte_carlo_samples"])
            draws = rng.random((samples, len(other))) < p
            changes = 0
            for row in draws:
                available = {item for item, flag in zip(other, row) if flag}
                changes += evaluate_rule(rule, available | {origin}) != evaluate_rule(rule, available)
            influence = changes / samples
        influences[origin] = float(influence)
    return influences


def effective_origins(record, config, claim_id=""):
    influences = origin_influences(record, config, claim_id, record["id"])
    threshold = config["inference"]["influence_cutoff"]
    return {origin for origin, value in influences.items() if value >= threshold}


def pair_labels(claim_record, config):
    providers = claim_record["providers"]
    active = {str(provider["id"]): effective_origins(provider, config, str(claim_record["id"])) for provider in providers}
    structural = {str(provider["id"]): {canonical_origin(x) for x in provider["referenced_origins"]} for provider in providers}
    labels = {}
    for left, right in itertools.combinations(providers, 2):
        a, b = str(left["id"]), str(right["id"])
        labels[(a, b)] = {"functional": int(bool(active[a] & active[b])), "structural": int(bool(structural[a] & structural[b]))}
    return labels
