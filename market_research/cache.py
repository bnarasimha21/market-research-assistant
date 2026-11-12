"""Disk cache for fetched pages and final results so repeats are free.

Two independent caches, both plain JSON files under .cache/ (gitignored):
pages (key: URL, TTL default 24h) and results (key: a per-field hash of
normalized names plus a config hash, TTL also 24h -- see
`result_cache_key` for why this changed from a raw joined string with no
expiry).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
PAGE_TTL_SECONDS = 24 * 60 * 60
# A cached comparison result can go stale for reasons the config hash
# doesn't capture (a competitor's real-world pricing page changes, a
# provider silently updates a model behind the same model id) even though
# nothing about the pipeline's own config changed. Reusing the same 24h
# window as the page cache bounds that staleness instead of a result being
# served forever (the previous behaviour, `ttl_seconds=None`) until someone
# notices and manually clears .cache/.
RESULT_TTL_SECONDS = 24 * 60 * 60


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


class DiskCache:
    def __init__(self, namespace: str, ttl_seconds: float | None = None, root: Path = CACHE_DIR) -> None:
        self.dir = root / namespace
        self.ttl_seconds = ttl_seconds

    def _path(self, key: str) -> Path:
        return self.dir / f"{_hash_key(key)}.json"

    def get(self, key: str) -> Any | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if self.ttl_seconds is not None:
            age = time.time() - payload.get("_cached_at", 0)
            if age > self.ttl_seconds:
                return None
        return payload.get("value")

    def set(self, key: str, value: Any) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self._path(key)
        payload = {"_cached_at": time.time(), "value": value}
        path.write_text(json.dumps(payload), encoding="utf-8")

    def clear(self) -> None:
        if self.dir.exists():
            for item in self.dir.glob("*.json"):
                item.unlink(missing_ok=True)


def pages_cache() -> DiskCache:
    return DiskCache("pages", ttl_seconds=PAGE_TTL_SECONDS)


def results_cache() -> DiskCache:
    return DiskCache("results", ttl_seconds=RESULT_TTL_SECONDS)


def result_cache_key(my_product: str, competitors: list[str], config_hash: str) -> str:
    """Build a collision-resistant cache key for one (my_product,
    competitors, config) combination.

    The previous version joined every field into one "|"-separated string
    before hashing: `"|".join([my_product, *competitors])`. Because "|" can
    appear inside a product/competitor name, that join is ambiguous --
    my_product="a|b", competitors=["c"] produces the exact same joined
    string ("a|b|c") as my_product="a", competitors=["b|c"], and
    competitors=["c"] (one entry containing no pipe) collides with
    competitors=["b", "c"] losing the "a" vs "a|b" distinction once both
    sides are flattened the same way. Hashing each field to a fixed-width
    sha256 hex digest BEFORE joining removes the ambiguity: no field's
    content can ever be mistaken for a delimiter between two other fields,
    since the only things being joined are already-fixed-length hex
    strings, not the original free-form text.
    """
    fields = [my_product.strip().lower(), *(c.strip().lower() for c in competitors), config_hash]
    digest_of_fields = ":".join(_hash_key(field) for field in fields)
    # The field count is folded in too so a 2-field vs 3-field input can
    # never coincide even in some currently-unforeseen hash-joining edge
    # case; this is defense in depth on top of the fixed-width hashing above.
    return _hash_key(f"{len(fields)}:{digest_of_fields}")
