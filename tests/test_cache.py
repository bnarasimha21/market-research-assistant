"""Unit tests for market_research.cache.DiskCache."""

from __future__ import annotations

import time

from market_research.cache import DiskCache, result_cache_key


def test_set_then_get_round_trips(tmp_path):
    cache = DiskCache("pages", ttl_seconds=None, root=tmp_path)
    cache.set("https://example.com/pricing", {"text": "hello"})
    assert cache.get("https://example.com/pricing") == {"text": "hello"}


def test_missing_key_returns_none(tmp_path):
    cache = DiskCache("pages", ttl_seconds=None, root=tmp_path)
    assert cache.get("https://example.com/nope") is None


def test_expired_entry_returns_none(tmp_path):
    cache = DiskCache("pages", ttl_seconds=0.01, root=tmp_path)
    cache.set("k", {"v": 1})
    time.sleep(0.05)
    assert cache.get("k") is None


def test_clear_removes_entries(tmp_path):
    cache = DiskCache("pages", ttl_seconds=None, root=tmp_path)
    cache.set("k1", {"v": 1})
    cache.set("k2", {"v": 2})
    cache.clear()
    assert cache.get("k1") is None
    assert cache.get("k2") is None


def test_result_cache_key_normalizes_case_and_whitespace():
    key1 = result_cache_key("Narsi", ["Acme", "Globex"], "cfg")
    key2 = result_cache_key("  narsi  ", ["  ACME  ", "globex"], "cfg")
    assert key1 == key2


def test_result_cache_key_differs_on_config():
    key1 = result_cache_key("Narsi", ["Acme"], "cfg1")
    key2 = result_cache_key("Narsi", ["Acme"], "cfg2")
    assert key1 != key2
