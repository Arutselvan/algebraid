"""
Model adapters for querying language-model APIs.

Each adapter implements ``run_tasks`` to execute a ``TaskSet`` against a
specific provider and return a ``{task_id: response}`` dictionary.

All provider SDKs are optional dependencies:

    openai       pip install -e '.[openai]'      OpenAI chat completion API.
    anthropic    pip install -e '.[anthropic]'   Anthropic Messages API.
    openrouter   pip install -e '.[openai]'      OpenRouter (500+ models, one key).
    custom_http  pip install -e '.[openai]'      Any OpenAI-compatible endpoint.
    huggingface  Not yet implemented; use custom_http with a vLLM/TGI server.
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

    Requires the ``openai`` package (``pip install -e '.[openai]'``)
    and the ``OPENAI_API_KEY`` environment variable.
    """

    def run_tasks(self, task_set: TaskSet) -> Dict[str, str]:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for this adapter. "
                "Install it with: pip install -e '.[openai]'"
            ) from None

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
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "The 'anthropic' package is required for this adapter. "
                "Install it with: pip install -e '.[anthropic]'"
            ) from None

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

    Requires the ``openai`` package (``pip install -e '.[openai]'``), used as
    the HTTP client only - no OpenAI API key is needed for self-hosted servers.
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
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for the custom_http adapter. "
                "Install it with: pip install -e '.[openai]'"
            ) from None

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


class OpenRouterAdapter(BaseAdapter):
    """Adapter for OpenRouter (https://openrouter.ai).

    Gives access to 500+ models — including GPT-4o, Claude, Gemini,
    DeepSeek-R1, Llama, Mistral, and more — through a single API key and
    an OpenAI-compatible interface.

    Requirements
    ------------
    * ``openai`` package: ``pip install -e '.[openai]'``
    * ``OPENROUTER_API_KEY`` environment variable set to your key.

    Model identifiers use the ``provider/name`` format that OpenRouter uses,
    e.g. ``deepseek/deepseek-r1``, ``google/gemini-flash-1.5``,
    ``meta-llama/llama-3.3-70b-instruct``.

    Optional environment variables
    -------------------------------
    ``OPENROUTER_SITE_URL``
        Appears as the HTTP-Referer header; shown in your OpenRouter dashboard.
        Defaults to ``https://github.com/Arutselvan/algebraid``.
    ``OPENROUTER_SITE_NAME``
        Shown alongside the URL in the dashboard. Defaults to ``ALGEBRAID``.

    Reasoning models (e.g. ``deepseek/deepseek-r1``)
    -------------------------------------------------
    When a model returns its chain-of-thought in a separate ``reasoning``
    field (OpenRouter's extended response format), the adapter prepends it
    as ``<think>…</think>`` before the content text so the verifier's
    ``_strip_think_blocks`` helper can cleanly separate scratchpad from
    the final answer.
    """

    _OPENROUTER_BASE = "https://openrouter.ai/api/v1"
    _DEFAULT_SITE_URL = "https://github.com/Arutselvan/algebraid"
    _DEFAULT_SITE_NAME = "ALGEBRAID"

    def run_tasks(self, task_set: TaskSet) -> Dict[str, str]:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for the openrouter adapter. "
                "Install it with: pip install -e '.[openai]'"
            ) from None

        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY is not set. "
                "Get a key at https://openrouter.ai/keys and export it as "
                "OPENROUTER_API_KEY=<your-key>."
            )

        site_url = os.environ.get("OPENROUTER_SITE_URL", self._DEFAULT_SITE_URL)
        site_name = os.environ.get("OPENROUTER_SITE_NAME", self._DEFAULT_SITE_NAME)

        client = OpenAI(
            base_url=self._OPENROUTER_BASE,
            api_key=api_key,
            default_headers={
                "HTTP-Referer": site_url,
                "X-Title": site_name,
            },
        )

        predictions: Dict[str, str] = {}
        total = len(task_set)

        if self.verbose:
            print(f"Running {total} tasks on OpenRouter model: {self.model} ...")

        for i, task in enumerate(task_set):
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": task.prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                msg = resp.choices[0].message
                content: str = msg.content or ""

                # Some reasoning models (e.g. deepseek/deepseek-r1) return the
                # chain-of-thought in a separate `reasoning` field rather than
                # inside <think> tags.  Wrap it so the verifier's
                # _strip_think_blocks helper works uniformly.
                reasoning: str = getattr(msg, "reasoning", None) or ""
                if reasoning and "<think>" not in content.lower():
                    content = f"<think>\n{reasoning}\n</think>\n{content}"

                predictions[task.task_id] = content
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
    "openrouter": OpenRouterAdapter,
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
