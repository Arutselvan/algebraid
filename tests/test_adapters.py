"""Unit tests for OllamaAdapter (mocked -- no running server required)."""

import json
import os
import urllib.error
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from algebraid.adapters import OllamaAdapter, get_adapter
from algebraid.task_model import Task


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(prompt: str = "What is 2+2?") -> Task:
    return Task(
        task_id="AG-test000001",
        prompt=prompt,
        answer="4",
        answer_raw="4",
        depth=1,
        family="intra",
        dimension="general",
        structures=["Z_5"],
        metadata={},
        solution_trace=[],
    )


def _mock_response(body: dict, code: int = 200) -> MagicMock:
    """Create a mock urllib response (context manager compatible)."""
    raw = json.dumps(body).encode()
    mock = MagicMock()
    mock.read.return_value = raw
    mock.__enter__ = lambda s: s
    mock.__exit__ = MagicMock(return_value=False)
    mock.status = code
    return mock


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestOllamaRegistration:
    def test_get_adapter_returns_ollama(self):
        assert get_adapter("ollama") is OllamaAdapter

    def test_get_adapter_case_insensitive(self):
        assert get_adapter("Ollama") is OllamaAdapter


# ---------------------------------------------------------------------------
# Constructor defaults
# ---------------------------------------------------------------------------

class TestOllamaConstructor:
    def test_default_base_url(self):
        adapter = OllamaAdapter(
            model="llama3.2:1b", temperature=0.0, max_tokens=256,
            delay=0.0, verbose=False,
        )
        assert adapter.base_url == "http://localhost:11434"

    def test_base_url_from_env(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", "http://myhost:5555")
        adapter = OllamaAdapter(
            model="llama3.2:1b", temperature=0.0, max_tokens=256,
            delay=0.0, verbose=False,
        )
        assert adapter.base_url == "http://myhost:5555"

    def test_base_url_kwarg_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", "http://envhost:1111")
        adapter = OllamaAdapter(
            model="llama3.2:1b", temperature=0.0, max_tokens=256,
            delay=0.0, verbose=False, base_url="http://kwarg:2222",
        )
        assert adapter.base_url == "http://kwarg:2222"

    def test_trailing_slash_stripped(self):
        adapter = OllamaAdapter(
            model="llama3.2:1b", temperature=0.0, max_tokens=256,
            delay=0.0, verbose=False, base_url="http://localhost:11434/",
        )
        assert adapter.base_url == "http://localhost:11434"


# ---------------------------------------------------------------------------
# _build_client
# ---------------------------------------------------------------------------

class TestOllamaBuildClient:
    def test_raises_when_server_unreachable(self):
        adapter = OllamaAdapter(
            model="llama3.2:1b", temperature=0.0, max_tokens=256,
            delay=0.0, verbose=False,
        )
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            with pytest.raises(ConnectionError, match="Is Ollama running"):
                adapter._build_client()

    def test_success_when_model_exists(self):
        adapter = OllamaAdapter(
            model="llama3.2:1b", temperature=0.0, max_tokens=256,
            delay=0.0, verbose=False,
        )
        version_resp = _mock_response({"version": "0.6.0"})
        tags_resp = _mock_response({
            "models": [{"name": "llama3.2:1b"}, {"name": "phi3:latest"}]
        })

        with patch("urllib.request.urlopen", side_effect=[version_resp, tags_resp]) as mock_open:
            result = adapter._build_client()

        assert result == adapter.base_url
        assert mock_open.call_count == 2  # version + tags, no pull

    def test_auto_pull_when_model_missing(self):
        adapter = OllamaAdapter(
            model="gemma:2b", temperature=0.0, max_tokens=256,
            delay=0.0, verbose=False,
        )
        version_resp = _mock_response({"version": "0.6.0"})
        tags_resp = _mock_response({"models": [{"name": "llama3.2:1b"}]})
        pull_resp = _mock_response({"status": "success"})

        with patch("urllib.request.urlopen", side_effect=[version_resp, tags_resp, pull_resp]):
            result = adapter._build_client()

        assert result == adapter.base_url


# ---------------------------------------------------------------------------
# _call_single
# ---------------------------------------------------------------------------

class TestOllamaCallSingle:
    def test_parses_valid_response(self):
        adapter = OllamaAdapter(
            model="llama3.2:1b", temperature=0.0, max_tokens=256,
            delay=0.0, verbose=False,
        )
        task = _make_task()
        chat_resp = _mock_response({
            "message": {"role": "assistant", "content": "The answer is 4."},
        })

        with patch("urllib.request.urlopen", return_value=chat_resp):
            result = adapter._call_single(adapter.base_url, task)

        assert result == "The answer is 4."

    def test_wraps_reasoning_in_think_tags(self):
        adapter = OllamaAdapter(
            model="deepseek-r1:7b", temperature=0.0, max_tokens=256,
            delay=0.0, verbose=False,
        )
        task = _make_task()
        chat_resp = _mock_response({
            "message": {
                "role": "assistant",
                "content": "Final Answer: 4",
                "reasoning": "Let me think step by step...",
            },
        })

        with patch("urllib.request.urlopen", return_value=chat_resp):
            result = adapter._call_single(adapter.base_url, task)

        assert result.startswith("<think>")
        assert "Let me think step by step..." in result
        assert "</think>" in result
        assert "Final Answer: 4" in result

    def test_no_think_tags_when_reasoning_absent(self):
        adapter = OllamaAdapter(
            model="llama3.2:1b", temperature=0.0, max_tokens=256,
            delay=0.0, verbose=False,
        )
        task = _make_task()
        chat_resp = _mock_response({
            "message": {"role": "assistant", "content": "Just the answer: 4"},
        })

        with patch("urllib.request.urlopen", return_value=chat_resp):
            result = adapter._call_single(adapter.base_url, task)

        assert "<think>" not in result
        assert result == "Just the answer: 4"

    def test_connection_error_message(self):
        adapter = OllamaAdapter(
            model="llama3.2:1b", temperature=0.0, max_tokens=256,
            delay=0.0, verbose=False,
        )
        task = _make_task()

        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            with pytest.raises(ConnectionError, match="Is Ollama running"):
                adapter._call_single(adapter.base_url, task)

    def test_404_suggests_pull(self):
        adapter = OllamaAdapter(
            model="nonexistent:latest", temperature=0.0, max_tokens=256,
            delay=0.0, verbose=False,
        )
        task = _make_task()

        exc = urllib.error.HTTPError(
            url="http://localhost:11434/api/chat",
            code=404,
            msg="Not Found",
            hdrs={},  # type: ignore[arg-type]
            fp=BytesIO(b""),
        )
        with patch("urllib.request.urlopen", side_effect=exc):
            with pytest.raises(RuntimeError, match="ollama pull"):
                adapter._call_single(adapter.base_url, task)
