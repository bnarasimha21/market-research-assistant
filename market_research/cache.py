"""Disk cache for fetched pages and final results so repeats are free.

Two independent caches, both plain JSON files under .cache/ (gitignored):
pages (key: URL, TTL default 24h) and results (key: normalized names +
config hash, no TTL -- a result is only replaced by a fresh run request,
never expires on its own since nothing about a product/competitor pairing
goes stale on a timer the way a fetched page does).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
PAGE_TTL_SECONDS = 24 * 60 * 60


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
    return DiskCache("results", ttl_seconds=None)


def result_cache_key(my_product: str, competitors: list[str], config_hash: str) -> str:
    normalized = "|".join([my_product.strip().lower(), *(c.strip().lower() for c in competitors)])
    return f"{normalized}::{config_hash}"
