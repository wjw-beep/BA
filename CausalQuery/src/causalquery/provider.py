from dataclasses import dataclass

from .hashing import canonical_origin, retain_origin, uniform_hash


@dataclass(frozen=True)
class ProbeRequest:
    question: str
    claim: str
    snapshot: str
    seed: str
    retention_probability: float


@dataclass(frozen=True)
class ProbeResponse:
    provider_id: str
    bit: int | None
    status: str
    snapshot: str


def rule_origins(rule):
    op = rule["op"]
    if op in {"any", "all", "at_least"}:
        return {canonical_origin(x) for x in rule.get("origins", [])}
    if op in {"any_of", "all_of"}:
        result = set()
        for child in rule["children"]:
            result.update(rule_origins(child))
        return result
    raise ValueError(f"Unknown support operator: {op}")


def validate_rule(rule):
    op = rule["op"]
    if op in {"any", "all", "at_least"}:
        origins = [canonical_origin(item) for item in rule.get("origins", [])]
        if not origins or len(set(origins)) != len(origins):
            raise ValueError("Support rules require nonempty distinct origin identifiers")
        if op == "at_least" and not 1 <= int(rule["count"]) <= len(origins):
            raise ValueError("at_least count must be between 1 and the number of origins")
    elif op in {"any_of", "all_of"}:
        children = rule.get("children", [])
        if not children:
            raise ValueError("Composite support rules require at least one child")
        for child in children:
            validate_rule(child)
    else:
        raise ValueError(f"Unknown support operator: {op}")


def evaluate_rule(rule, available):
    op = rule["op"]
    if op == "any":
        return any(canonical_origin(x) in available for x in rule["origins"])
    if op == "all":
        return all(canonical_origin(x) in available for x in rule["origins"])
    if op == "at_least":
        return sum(canonical_origin(x) in available for x in rule["origins"]) >= int(rule["count"])
    if op == "any_of":
        return any(evaluate_rule(child, available) for child in rule["children"])
    if op == "all_of":
        return all(evaluate_rule(child, available) for child in rule["children"])
    raise ValueError(f"Unknown support operator: {op}")


class InstrumentedProvider:
    def __init__(self, record, claim_record, namespace):
        self.identifier = str(record["id"])
        self.claim = str(claim_record["claim"])
        self.question = str(claim_record["question"])
        self.snapshot = str(claim_record["snapshot"])
        self.origins = frozenset(canonical_origin(x) for x in record["referenced_origins"])
        self.rule = record["support"]
        validate_rule(self.rule)
        self.failure_rate = float(record.get("failure_rate", 0))
        self.failure_seeds = frozenset(str(x) for x in record.get("unavailable_seeds", []))
        self.namespace = namespace
        if not self.origins or not 0 <= self.failure_rate <= 1:
            raise ValueError("Provider origins and failure rate are invalid")
        if not rule_origins(self.rule).issubset(self.origins):
            raise ValueError(f"Provider {self.identifier} references undeclared support origins")
        if not evaluate_rule(self.rule, self.origins):
            raise ValueError(f"Provider {self.identifier} is not initially supporting its claim")

    def respond(self, request):
        if request.snapshot != self.snapshot or request.claim != self.claim or request.question != self.question:
            return ProbeResponse(self.identifier, None, "unavailable", self.snapshot)
        if request.seed in self.failure_seeds or uniform_hash(self.namespace + "/availability", request.seed, self.identifier) < self.failure_rate:
            return ProbeResponse(self.identifier, None, "unavailable", self.snapshot)
        available = {origin for origin in self.origins if retain_origin(request.seed, origin, request.retention_probability, self.namespace + "/mask")}
        bit = int(evaluate_rule(self.rule, available))
        return ProbeResponse(self.identifier, bit, "ok", self.snapshot)
