import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import analyzer
import providers
from data import COMPLAINTS

GOOD = {"category": "Live wire", "urgency": "Critical", "department": "Electricity",
        "next_action": "Dispatch crew to isolate the line", "reason": "Risk to children", "confidence": 0.95}


class FakeLLM(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible server: /v1/chat/completions and Ollama's /api/tags."""
    requests_seen = []

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"models": []}')

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeLLM.requests_seen.append((self.path, self.headers.get("Authorization"), body))
        reply = {"choices": [{"message": {"content": "Sure!\n" + json.dumps(GOOD)}}]}
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps(reply).encode())

    def log_message(self, *a):
        pass


@pytest.fixture
def fake_server(monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), FakeLLM)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}"
    FakeLLM.requests_seen = []
    # Point every OpenAI-compatible provider at the fake server.
    for name, cfg in list(providers.OPENAI_COMPATIBLE.items()):
        monkeypatch.setitem(providers.OPENAI_COMPATIBLE, name, (cfg[0], url + "/v1", *cfg[2:]))
    monkeypatch.setenv("OLLAMA_URL", url)
    yield url
    server.shutdown()


def test_rules_fallback():
    r = analyzer.classify("live wire sparking near school", "rules")
    assert (r["urgency"], r["department"], r["source"]) == ("Critical", "Electricity", "rules")
    assert analyzer.classify("Broken bench in park, minor", "rules")["urgency"] == "Low"


def test_unconfigured_provider_falls_back(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert analyzer.classify("live wire sparking", "openai")["source"] == "rules"


def test_parse_json_with_chatter():
    assert providers.parse_json('Here you go:\n```json\n{"a": 1}\n```') == {"a": 1}


@pytest.mark.parametrize("name", list(providers.OPENAI_COMPATIBLE))
def test_every_openai_compatible_provider(fake_server, monkeypatch, name):
    key_env = providers.OPENAI_COMPATIBLE[name][2]
    if key_env:
        monkeypatch.setenv(key_env, "test-key")
    r = analyzer.classify("live wire sparking near school", name)
    assert r["source"] == name and r["urgency"] == "Critical"
    path, auth, body = FakeLLM.requests_seen[-1]
    assert path == "/v1/chat/completions"
    assert body["model"] == providers.model_for(name)
    assert auth == ("Bearer test-key" if key_env else None)


def test_gemma_default_model():
    assert providers.model_for("gemma") == "gemma4"


def test_claude_path(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setattr(providers, "_claude", lambda prompt: json.dumps(GOOD))
    assert analyzer.classify("live wire", "claude")["source"] == "claude"


def test_bad_values_are_clamped(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setattr(providers, "_claude", lambda p: json.dumps({**GOOD, "urgency": "URGENT!!", "department": "Mars"}))
    r = analyzer.classify("x", "claude")
    assert (r["urgency"], r["department"]) == ("Medium", "General")


def test_provider_error_falls_back(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setattr(providers, "_claude", lambda p: (_ for _ in ()).throw(RuntimeError("down")))
    r = analyzer.classify("live wire sparking", "claude")
    assert r["source"] == "rules" and "claude failed" in r["reason"]


def test_consensus_vote(fake_server, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    r = analyzer.consensus("live wire sparking near school", ["gemma", "llama", "openai"])
    assert r["urgency"] == "Critical" and r["confidence"] == 1.0
    assert r["source"] == "vote:gemma+llama+openai"


def test_duplicates_group_planted_pairs():
    clusters, _ = analyzer.find_duplicates(COMPLAINTS, use_embeddings=False)
    assert clusters[0] == clusters[1] == clusters[2] != -1   # Main St streetlight x3
    assert clusters[14] == clusters[15] != -1                # Wi-Fi vs wifi
    assert clusters[23] == -1                                # bench is unique


def test_elevenlabs(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    assert providers.speak("hi") is None
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")

    class R:
        content = b"ID3mp3"
        def raise_for_status(self): pass

    seen = {}
    monkeypatch.setattr(providers.requests, "post", lambda url, **kw: seen.update(url=url, **kw) or R())
    assert providers.speak("hello") == b"ID3mp3"
    assert "api.elevenlabs.io/v1/text-to-speech/" in seen["url"] and seen["headers"]["xi-api-key"] == "k"


def test_analyze_all():
    rows = analyzer.analyze_all(COMPLAINTS[:5], "rules")
    assert len(rows) == 5 and all({"urgency", "department", "cluster", "dedup"} <= r.keys() for r in rows)
