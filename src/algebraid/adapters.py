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

Checkpointing
-------------
``run_tasks`` accepts an optional ``checkpoint_path`` argument.  When set:

* Existing predictions are loaded from that file at startup so a crashed
  or interrupted run can be resumed without re-querying completed tasks.
* Predictions are written to that file every ``checkpoint_every`` tasks
  (default 10) and again at the end.

The CLI passes the final output path as ``checkpoint_path`` automatically,
so ``algebraid run`` and ``algebraid pipeline`` are both resume-safe.
"""

import json
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from .task_model import Task, TaskSet


class BaseAdapter(ABC):
    """Abstract base class for all model adapters."""

    def __init__(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        delay: float,
        verbose: bool,
        checkpoint_every: int = 10,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.delay = delay
        self.verbose = verbose
        self.checkpoint_every = checkpoint_every

    # ------------------------------------------------------------------
    # Subclass interface — implement these two methods per provider
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_client(self) -> Any:
        """Initialise and return the provider's API client object."""

    @abstractmethod
    def _call_single(self, client: Any, task: Task) -> str:
        """Make one API call and return the response text."""

    # ------------------------------------------------------------------
    # Shared loop with checkpoint support
    # ------------------------------------------------------------------

    def run_tasks(
        self,
        task_set: TaskSet,
        checkpoint_path: Optional[str] = None,
    ) -> Dict[str, str]:
        """Run all tasks and return ``{task_id: response_text}``.

        Args:
            task_set: The set of tasks to evaluate.
            checkpoint_path: Optional path to a JSON file used as a
                rolling checkpoint.  Existing entries are loaded and
                skipped; new entries are flushed every
                ``self.checkpoint_every`` completions.
        """
        # ── Resume from checkpoint ────────────────────────────────────
        predictions: Dict[str, str] = {}
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path) as f:
                    predictions = json.load(f)
                if predictions and self.verbose:
                    print(
                        f"  Resuming from checkpoint: "
                        f"{len(predictions)}/{len(task_set)} tasks already done."
                    )
            except (json.JSONDecodeError, OSError):
                predictions = {}

        pending = [t for t in task_set if t.task_id not in predictions]
        total = len(task_set)

        if not pending:
            if self.verbose:
                print("  All tasks already completed (loaded from checkpoint).")
            return predictions

        if self.verbose:
            already = total - len(pending)
            label = f"{self.model}"
            print(
                f"Running {len(pending)} task(s) on {label}"
                + (f" ({already} skipped — already done)" if already else "")
                + " ..."
            )

        client = self._build_client()

        for i, task in enumerate(pending):
            try:
                predictions[task.task_id] = self._call_single(client, task)
            except Exception as e:
                print(f"  Error on {task.task_id}: {e}")
                predictions[task.task_id] = "[ERROR]"

            done = total - len(pending) + i + 1
            if self.verbose and done % 10 == 0:
                print(f"  [{done}/{total}] completed")

            if checkpoint_path and (i + 1) % self.checkpoint_every == 0:
                self._save_checkpoint(predictions, checkpoint_path)

            if self.delay > 0:
                time.sleep(self.delay)

        if checkpoint_path:
            self._save_checkpoint(predictions, checkpoint_path)

        if self.verbose:
            print(f"Done. {len(predictions)} predictions collected.")
        return predictions

    @staticmethod
    def _save_checkpoint(predictions: Dict[str, str], path: str) -> None:
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(path, "w") as f:
            json.dump(predictions, f, indent=2)


class OpenAIAdapter(BaseAdapter):
    """Adapter for OpenAI and OpenAI-compatible chat completion APIs.

    Requires the ``openai`` package (``pip install -e '.[openai]'``)
    and the ``OPENAI_API_KEY`` environment variable.
    """

    def _build_client(self) -> Any:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for this adapter. "
                "Install it with: pip install -e '.[openai]'"
            ) from None
        return OpenAI()

    def _call_single(self, client: Any, task: Task) -> str:
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": task.prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content or ""


class AnthropicAdapter(BaseAdapter):
    """Adapter for Anthropic's Messages API.

    Requires the ``anthropic`` package and ``ANTHROPIC_API_KEY`` environment variable.

    Example model identifiers: ``claude-sonnet-4-6``, ``claude-haiku-4-5-20251001``.
    """

    def _build_client(self) -> Any:
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "The 'anthropic' package is required for this adapter. "
                "Install it with: pip install -e '.[anthropic]'"
            ) from None
        return Anthropic()

    def _call_single(self, client: Any, task: Task) -> str:
        resp = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": task.prompt}],
        )
        return resp.content[0].text


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

    def _build_client(self) -> Any:
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

        return OpenAI(
            base_url=self._OPENROUTER_BASE,
            api_key=api_key,
            default_headers={
                "HTTP-Referer": site_url,
                "X-Title": site_name,
            },
        )

    def _call_single(self, client: Any, task: Task) -> str:
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

        return content


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

    def _build_client(self) -> Any:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for the custom_http adapter. "
                "Install it with: pip install -e '.[openai]'"
            ) from None

        return OpenAI(
            base_url=self.base_url,
            api_key=os.environ.get("OPENAI_API_KEY", "none"),
        )

    def _call_single(self, client: Any, task: Task) -> str:
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": task.prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content or ""


class HuggingFaceAdapter(BaseAdapter):
    """Adapter for locally hosted HuggingFace models.

    Not yet implemented. To use a local HuggingFace model, serve it with
    ``text-generation-inference`` or ``vLLM`` and point ``custom_http``
    at the resulting OpenAI-compatible endpoint instead.
    """

    def _build_client(self) -> Any:
        raise NotImplementedError(
            "HuggingFaceAdapter is not yet implemented. "
            "To evaluate a local HuggingFace model, serve it with vLLM or "
            "text-generation-inference and use --adapter custom_http with "
            "ALGEBRAID_API_BASE pointing at the server's /v1 endpoint."
        )

    def _call_single(self, client: Any, task: Task) -> str:
        raise NotImplementedError


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
