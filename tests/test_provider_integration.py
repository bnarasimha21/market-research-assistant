"""Integration test: LLM_PROVIDER=do_serverless pointed at a throwaway local
HTTP server speaking the OpenAI chat-completions shape, asserting the real
request path, Authorization header and model id -- not mocking crewai.LLM
itself, so this proves get_llm() really builds a working OpenAI-compatible
client against DigitalOcean's serverless inference endpoint shape.
"""

from __future__ import annotations

import http.server
import json
import threading

import pytest

from market_research.llm import get_llm

pytestmark = pytest.mark.integration


class _RecordingHandler(http.server.BaseHTTPRequestHandler):
    requests: list[dict] = []

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        _RecordingHandler.requests.append(
            {
                "path": self.path,
                "authorization": self.headers.get("Authorization"),
                "body": body,
            }
        )
        response = {
            "id": "fake-completion",
            "object": "chat.completion",
            "created": 0,
            "model": body.get("model", "unknown"),
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "ack"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        }
        encoded = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *args) -> None:
        pass


@pytest.fixture
def fake_do_server():
    _RecordingHandler.requests = []
    server = http.server.HTTPServer(("127.0.0.1", 0), _RecordingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_do_serverless_llm_hits_the_recorded_server_with_correct_shape(fake_do_server, monkeypatch):
    port = fake_do_server
    monkeypatch.setenv("LLM_PROVIDER", "do_serverless")
    monkeypatch.setenv("DO_MODEL_ACCESS_KEY", "do-test-key")
    monkeypatch.setenv("DO_MODEL_ID", "llama-3.1-70b-instruct")

    llm = get_llm("pipeline", base_url_override=f"http://127.0.0.1:{port}/v1")
    result = llm.call("ping the fake DigitalOcean endpoint")

    assert result == "ack"
    assert len(_RecordingHandler.requests) == 1
    recorded = _RecordingHandler.requests[0]
    assert recorded["path"] == "/v1/chat/completions"
    assert recorded["authorization"] == "Bearer do-test-key"
    assert recorded["body"]["model"] == "llama-3.1-70b-instruct"
