from .hashing import seed_for_probe
from .provider import InstrumentedProvider, ProbeRequest


def run_provider_batch(claim, providers, config):
    protocol = config["protocol"]
    identifiers = [str(provider.identifier) for provider in providers]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Duplicate provider identifiers within claim")
    matrix = {identifier: [] for identifier in identifiers}
    seeds = []
    for index in range(int(protocol["probes"])):
        seed = seed_for_probe(protocol["master_seed"], claim["id"], index, protocol["namespace"])
        seeds.append(seed)
        request = ProbeRequest(str(claim["question"]), str(claim["claim"]), str(claim["snapshot"]), seed, float(protocol["retention_probability"]))
        for provider in providers:
            try:
                response = provider.respond(request)
                if response.provider_id == provider.identifier and response.status == "ok" and response.snapshot == request.snapshot and type(response.bit) is int and response.bit in (0, 1):
                    payload = {"status": "ok", "bit": response.bit, "snapshot": response.snapshot}
                else:
                    payload = {"status": "unavailable", "bit": None, "snapshot": response.snapshot}
            except Exception:
                payload = {"status": "unavailable", "bit": None, "snapshot": request.snapshot}
            matrix[provider.identifier].append(payload)
    return {"claim_id": str(claim["id"]), "snapshot": str(claim["snapshot"]), "seeds": seeds, "responses": matrix}


def run_claim_probes(claim_record, config):
    providers = [InstrumentedProvider(item, claim_record, config["protocol"]["namespace"]) for item in claim_record["providers"]]
    return run_provider_batch(claim_record, providers, config)
