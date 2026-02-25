"""
COMPEVAL LLM Runner - Run benchmark tasks against LLM APIs.

Supports any OpenAI-compatible API endpoint. Handles rate limiting,
retries, and batched execution.
"""

import os
import time
import json
from typing import Dict, List, Optional
from .task_model import Task, TaskSet

SYSTEM_PROMPT = (
    "You are a precise mathematical calculator. You will be given algebraic "
    "problems involving group theory, finite fields, and compositional operations. "
    "Work each problem step by step, then provide ONLY the final answer on the "
    "last line. Do not include any explanation after the final answer."
)

def run_tasks_on_llm(
    task_set: TaskSet,
    model: str = "gpt-4.1-nano",
    max_tasks: Optional[int] = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    delay: float = 0.5,
    verbose: bool = True,
) -> Dict[str, str]:
    """
    Run all tasks in a TaskSet against an LLM and return predictions.

    Args:
        task_set: The TaskSet to evaluate.
        model: Model name for the API.
        max_tasks: Maximum number of tasks to run (None = all).
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in the response.
        delay: Delay between API calls in seconds.
        verbose: Whether to print progress.

    Returns:
        Dict mapping task_id -> model response string.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Please install openai: pip install openai")

    client = OpenAI()
    predictions: dict[str, str] = {}
    total: int = len(task_set) if not max_tasks else min(max_tasks, len(task_set))
    tasks: list = list(task_set)[:total]

    if verbose:
        print(f"Running {total} tasks on {model}...")

    for i, task in enumerate(tasks):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": task.prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            answer = response.choices[0].message.content.strip()
            predictions[task.task_id] = answer

            if verbose and (i + 1) % 10 == 0:
                print(f"  [{i+1}/{total}] completed")

        except Exception as e:
            if verbose:
                print(f"  [{i+1}/{total}] ERROR: {e}")
            predictions[task.task_id] = "<error>"

        if delay > 0:
            time.sleep(delay)

    if verbose:
        print(f"Done. {len(predictions)} predictions collected.")

    return predictions

