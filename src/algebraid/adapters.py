"""
Model adapters for querying language-model APIs.

Each adapter implements ``run_tasks`` to execute a ``TaskSet`` against a
specific provider and return a ``{task_id: response}`` dictionary.

Supported adapters
------------------
openai       OpenAI and OpenAI-compatible chat completion APIs.
anthropic    Anthropic Messages API (requires ``anthropic`` package).
custom_http  Any OpenAI-compatible endpoint via a custom base URL.
             Set ALGEBRAID_API_BASE or pass base_url as a kwarg.
huggingface  Not yet implemented (requires local model setup).
"""

import os
import time
from abc import ABC, abstractmethod
from typing import Dict, Any

from .task_model import TaskSet


class BaseAdapter(ABC):
    """Abstract base class for all model adapters."""

    def __init__(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        delay: float,
        verbose: bool,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.delay = delay
        self.verbose = verbose

    @abstractmethod
    def run_tasks(self, task_set: TaskSet) -> Dict[str, str]:
        """Run all tasks and return ``{task_id: response_text}``."""

    def _log(self, i: int, total: int) -> None:
        if self.verbose and (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{total}] completed")


class OpenAIAdapter(BaseAdapter):
    """Adapter for OpenAI and OpenAI-compatible chat completion APIs.

    Requires the ``openai`` package and ``OPENAI_API_KEY`` environment variable.
    """

    def run_tasks(self, task_set: TaskSet) -> Dict[str, str]:
        from openai import OpenAI

        client = OpenAI()
        predictions: Dict[str, str] = {}
        total = len(task_set)

        if self.verbose:
            print(f"Running {total} tasks on OpenAI model: {self.model} ...")

        for i, task in enumerate(task_set):
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": task.prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                predictions[task.task_id] = resp.choices[0].message.content or ""
            except Exception as e:
                print(f"  Error on {task.task_id}: {e}")
                predictions[task.task_id] = "[ERROR]"

            self._log(i, total)
            if self.delay > 0:
                time.sleep(self.delay)

        if self.verbose:
            print(f"Done. {len(predictions)} predictions collected.")
        return predictions


class AnthropicAdapter(BaseAdapter):
    """Adapter for Anthropic's Messages API.

    Requires the ``anthropic`` package and ``ANTHROPIC_API_KEY`` environment variable.

    Example model identifiers: ``claude-sonnet-4-6``, ``claude-haiku-4-5-20251001``.
    """

    def run_tasks(self, task_set: TaskSet) -> Dict[str, str]:
        from anthropic import Anthropic

        client = Anthropic()
        predictions: Dict[str, str] = {}
        total = len(task_set)

        if self.verbose:
            print(f"Running {total} tasks on Anthropic model: {self.model} ...")

        for i, task in enumerate(task_set):
            try:
                resp = client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": task.prompt}],
                )
                predictions[task.task_id] = resp.content[0].text
            except Exception as e:
                print(f"  Error on {task.task_id}: {e}")
                predictions[task.task_id] = "[ERROR]"

            self._log(i, total)
            if self.delay > 0:
                time.sleep(self.delay)

        if self.verbose:
            print(f"Done. {len(predictions)} predictions collected.")
        return predictions


class CustomHTTPAdapter(BaseAdapter):
    """Adapter for any OpenAI-compatible HTTP endpoint.

    Works with self-hosted servers such as vLLM, Ollama, LM Studio, or
    any proxy that exposes the ``/v1/chat/completions`` endpoint.

    The base URL is resolved in this order:
      1. ``ALGEBRAID_API_BASE`` environment variable.
      2. The ``base_url`` kwarg passed to the constructor (not exposed via CLI).
      3. Falls back to ``http://localhost:11434/v1`` (Ollama default).

    Requires the ``openai`` package (used as HTTP client only; no OpenAI key needed).
    Set ``OPENAI_API_KEY=none`` if the server does not require authentication.
    """

    def __init__(self, *args: Any, base_url: str = "", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.base_url = (
            base_url
            or os.environ.get("ALGEBRAID_API_BASE", "")
            or "http://localhost:11434/v1"
        )

    def run_tasks(self, task_set: TaskSet) -> Dict[str, str]:
        from openai import OpenAI

        client = OpenAI(
            base_url=self.base_url,
            api_key=os.environ.get("OPENAI_API_KEY", "none"),
        )
        predictions: Dict[str, str] = {}
        total = len(task_set)

        if self.verbose:
            print(f"Running {total} tasks on {self.model} at {self.base_url} ...")

        for i, task in enumerate(task_set):
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": task.prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                predictions[task.task_id] = resp.choices[0].message.content or ""
            except Exception as e:
                print(f"  Error on {task.task_id}: {e}")
                predictions[task.task_id] = "[ERROR]"

            self._log(i, total)
            if self.delay > 0:
                time.sleep(self.delay)

        if self.verbose:
            print(f"Done. {len(predictions)} predictions collected.")
        return predictions


class HuggingFaceAdapter(BaseAdapter):
    """Adapter for locally hosted HuggingFace models.

    Not yet implemented. To use a local HuggingFace model, serve it with
    ``text-generation-inference`` or ``vLLM`` and point ``custom_http``
    at the resulting OpenAI-compatible endpoint instead.
    """

    def run_tasks(self, task_set: TaskSet) -> Dict[str, str]:
        raise NotImplementedError(
            "HuggingFaceAdapter is not yet implemented. "
            "To evaluate a local HuggingFace model, serve it with vLLM or "
            "text-generation-inference and use --adapter custom_http with "
            "ALGEBRAID_API_BASE pointing at the server's /v1 endpoint."
        )


ADAPTER_MAP: Dict[str, type] = {
    "openai": OpenAIAdapter,
    "anthropic": AnthropicAdapter,
    "custom_http": CustomHTTPAdapter,
    "huggingface": HuggingFaceAdapter,
}


def get_adapter(name: str) -> type:
    """Return an adapter class by name (case-insensitive)."""
    cls = ADAPTER_MAP.get(name.lower())
    if cls is None:
        raise ValueError(
            f"Unknown adapter {name!r}. Available: {list(ADAPTER_MAP)}"
        )
    return cls
