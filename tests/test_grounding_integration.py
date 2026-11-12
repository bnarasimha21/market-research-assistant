"""Integration test: competitor stage against a real local HTTP server
serving two recorded HTML pages, with a FakeLLM scripted to return one real
quote and one invented quote. Only the real quote should survive the
deterministic grounding check -- never an LLM's word for it.
"""

from __future__ import annotations

import http.server
import json
import threading
from datetime import UTC

import pytest
from crewai import Crew

from market_research.agents import build_competitor_agent, build_competitor_task
from market_research.flow import _parse_and_ground_competitor_facts
from market_research.sources import FixtureSearch, gather_pages
from tests.conftest import FakeLLM

pytestmark = pytest.mark.integration

PAGES = {
    "/pricing": "<html><body><h1>Pricing</h1><p>Get started in minutes with our free tier.</p></body></html>",
    "/docs": "<html><body><h1>Docs</h1><p>See our full API reference for integration details.</p></body></html>",
}


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - stdlib-mandated name
        body = PAGES.get(self.path)
        if body is None:
            self.send_response(404)
            self.end_headers()
            return
        encoded = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *args) -> None:  # silence test output
        pass


@pytest.fixture
def local_server():
    server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_one_real_and_one_invented_quote_only_real_one_survives(local_server, monkeypatch):
    import market_research.sources as sources_mod

    port = local_server
    pricing_url = f"https://127.0.0.1:{port}/pricing"
    docs_url = f"https://127.0.0.1:{port}/docs"

    # fetch_page is https-only by contract; http.server only speaks plain
    # http. So this test fetches over a real socket on the downgraded
    # http:// URL (a genuine end-to-end fetch, not a mock), then re-presents
    # the result under the https:// URL the competitor stage was told
    # about, to exercise the real _strip_html/Page path without needing a
    # local TLS certificate.
    def patched_fetch_page(url: str, use_cache: bool = True):
        from datetime import datetime

        import requests

        from market_research.sources import Page, _strip_html

        downgraded = url.replace("https://", "http://", 1)
        response = requests.get(downgraded, timeout=5)
        response.raise_for_status()
        return Page(url=url, fetched_at=datetime.now(UTC).isoformat(), text=_strip_html(response.text))

    monkeypatch.setattr(sources_mod, "fetch_page", patched_fetch_page)

    search = FixtureSearch({"Acme pricing features documentation": [pricing_url, docs_url]})
    pages = gather_pages("Acme", search)
    assert len(pages.pages) == 2

    competitor_reply = json.dumps(
        {
            "items": [
                {
                    "claim": "Free tier available",
                    "source_url": pricing_url,
                    "quote": "Get started in minutes with our free tier",
                },
                {
                    "claim": "Invented claim",
                    "source_url": pricing_url,
                    "quote": "this exact sentence never appears on the page",
                },
            ]
        }
    )
    fake = FakeLLM(replies={"Competitor Analyst": competitor_reply})
    agent = build_competitor_agent(fake)
    task = build_competitor_task(agent, "Acme", pages)
    Crew(agents=[agent], tasks=[task]).kickoff()

    facts, dropped = _parse_and_ground_competitor_facts(task.output.raw, pages)
    assert dropped == 1
    assert len(facts) == 1
    assert facts[0].claim == "Free tier available"
    assert facts[0].source_url == pricing_url
