"""
Model adapters for running ALGEBRAID tasks on various LLM providers.

Each adapter must implement a `run_tasks` function that takes a TaskSet and
model parameters, and returns a dictionary of predictions.
"""

import os
import time
import json
from abc import ABC, abstractmethod
from typing import Dict, Any

from .task_model import TaskSet

class BaseAdapter(ABC):
    """Abstract base class for all model adapters."""

    def __init__(self, model: str, temperature: float, max_tokens: int, delay: float, verbose: bool):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.delay = delay
        self.verbose = verbose

    @abstractmethod
    def run_tasks(self, task_set: TaskSet) -> Dict[str, Any]:
        """Run a TaskSet and return a dictionary of predictions."""
        pass


class OpenAIAdapter(BaseAdapter):
    """Adapter for OpenAI-compatible APIs."""

    def run_tasks(self, task_set: TaskSet) -> Dict[str, Any]:
        from openai import OpenAI

        client = OpenAI()
        predictions = {}
        total = len(task_set)

        if self.verbose:
            print(f"Running {total} tasks on OpenAI model: {self.model}...")

        for i, task in enumerate(task_set):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": task.prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                prediction = response.choices[0].message.content
                predictions[task.task_id] = prediction

                if self.verbose and (i + 1) % 10 == 0:
                    print(f"  [{i+1}/{total}] completed")

            except Exception as e:
                print(f"Error on task {task.task_id}: {e}")
                predictions[task.task_id] = "[ERROR]"

            if self.delay > 0:
                time.sleep(self.delay)

        if self.verbose:
            print(f"Done. {len(predictions)} predictions collected.")

        return predictions


class AnthropicAdapter(BaseAdapter):
    """Adapter for Anthropic APIs."""

    def run_tasks(self, task_set: TaskSet) -> Dict[str, Any]:
        # Placeholder for Anthropic implementation
        print("Anthropic adapter is not yet implemented.")
        return {}


class HuggingFaceAdapter(BaseAdapter):
    """Adapter for HuggingFace models."""

    def run_tasks(self, task_set: TaskSet) -> Dict[str, Any]:
        # Placeholder for HuggingFace implementation
        print("HuggingFace adapter is not yet implemented.")
        return {}


class CustomHTTPAdapter(BaseAdapter):
    """Adapter for custom HTTP endpoints."""

    def run_tasks(self, task_set: TaskSet) -> Dict[str, Any]:
        # Placeholder for custom HTTP implementation
        print("Custom HTTP adapter is not yet implemented.")
        return {}


ADAPTER_MAP = {
    "openai": OpenAIAdapter,
    "anthropic": AnthropicAdapter,
    "huggingface": HuggingFaceAdapter,
    "custom_http": CustomHTTPAdapter,
}

def get_adapter(name: str):
    """Factory function to get an adapter by name."""
    adapter_class = ADAPTER_MAP.get(name.lower())
    if not adapter_class:
        raise ValueError(f"Unknown adapter: {name}. Available: {list(ADAPTER_MAP.keys())}")
    return adapter_class
