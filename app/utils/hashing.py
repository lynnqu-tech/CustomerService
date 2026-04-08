from hashlib import sha256


def sha256_text(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()
