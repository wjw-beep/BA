import hashlib
import json
import unicodedata


def canonical_origin(origin):
    if not isinstance(origin, str):
        raise ValueError("Origin identifiers must be strings")
    result = unicodedata.normalize("NFC", origin).strip()
    if not result:
        raise ValueError("Origin identifier cannot be empty")
    return result


def uniform_hash(namespace, *parts):
    payload = json.dumps([namespace, *parts], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    value = hashlib.blake2b(payload, digest_size=16).digest()
    return int.from_bytes(value, "big") / (1 << 128)


def retain_origin(seed, origin, probability, namespace):
    return uniform_hash(namespace, str(seed), canonical_origin(origin)) < probability


def seed_for_probe(master_seed, claim_id, number, namespace):
    payload = json.dumps([namespace, str(master_seed), str(claim_id), int(number)], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
