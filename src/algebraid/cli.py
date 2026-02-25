"""
Command-line interface for ALGEBRAID.

Commands
--------
generate   Create a new task set.
run        Execute a task set against a language model.
evaluate   Score model predictions against a task set.
validate   Check a task set for structural or semantic issues.
"""

import argparse
import os
import json
import re
import time
from datetime import date
from pathlib import Path

from .generator import AlgebraidGenerator
from .evaluator import AlgebraidEvaluator
from .task_model import TaskSet
from .adapters import get_adapter
from .tasks.validator import TaskValidator, print_report


# ── naming helpers ─────────────────────────────────────────────────────────

def _default_task_path(seed: int) -> str:
    """Default task-set path: ``./data/algebraid_s{seed}_{YYYYMMDD}.jsonl``."""
    return f"./data/algebraid_s{seed}_{date.today():%Y%m%d}.jsonl"


def _sanitize(name: str) -> str:
    """Turn an arbitrary string into a filesystem-safe slug."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()


def _stem(path: str) -> str:
    """Extract the filename stem (no directory, no extension)."""
    return Path(path).stem


def _default_predictions_path(task_set_path: str, model: str) -> str:
    """Default predictions path encoding the task set and model.

    Example: ``./results/preds_gpt_4_1_nano_algebraid_s42_20260225.json``
    """
    ts = _sanitize(_stem(task_set_path))
    m = _sanitize(model)
    return f"./results/preds_{m}_{ts}.json"


def _default_report_path(task_set_path: str, model_name: str) -> str:
    """Default evaluation report path encoding the task set and model.

    Example: ``./results/report_gpt_4_1_nano_algebraid_s42_20260225.json``
    """
    ts = _sanitize(_stem(task_set_path))
    m = _sanitize(model_name)
    return f"./results/report_{m}_{ts}.json"


# ── generate ───────────────────────────────────────────────────────────────

def _generate(args: argparse.Namespace) -> None:
    output = args.output or _default_task_path(args.seed)
    print(f"Generating task set (seed={args.seed}) ...")

    gen = AlgebraidGenerator(seed=args.seed)
    task_set = gen.generate(
        depths=args.depths,
        tasks_per_depth=args.tasks_per_depth,
        families=args.families,
        include_dimensions=not args.no_dims,
    )

    out_dir = os.path.dirname(output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    task_set.to_jsonl(output)
    print(f"Saved {len(task_set)} tasks to {output}")
    print(task_set.summary())

    if not args.skip_validation:
        report = TaskValidator().validate_set(task_set)
        print_report(report)
        if report["failed"] > 0:
            print(f"WARNING: {report['failed']} task(s) failed validation.")


# ── run ────────────────────────────────────────────────────────────────────

def _run(args: argparse.Namespace) -> None:
    if not os.path.exists(args.task_set):
        print(f"Error: file not found — {args.task_set}")
        return

    task_set = TaskSet.from_jsonl(args.task_set)
    output = args.output or _default_predictions_path(args.task_set, args.model)
    print(f"Running {len(task_set)} tasks on {args.model} ...")

    Adapter = get_adapter(args.adapter)
    adapter = Adapter(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        delay=args.delay,
        verbose=not args.quiet,
    )

    t0 = time.time()
    predictions = adapter.run_tasks(task_set)
    print(f"Completed in {time.time() - t0:.1f}s")

    out_dir = os.path.dirname(output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"Predictions saved to {output}")


# ── evaluate ───────────────────────────────────────────────────────────────

def _evaluate(args: argparse.Namespace) -> None:
    for path, label in [(args.task_set, "Task set"), (args.predictions, "Predictions")]:
        if not os.path.exists(path):
            print(f"Error: {label} not found — {path}")
            return

    task_set = TaskSet.from_jsonl(args.task_set)
    with open(args.predictions) as f:
        predictions = json.load(f)

    evaluator = AlgebraidEvaluator(strict=args.strict)
    report = evaluator.evaluate(task_set, predictions, model_name=args.model_name)

    output = args.output or _default_report_path(args.task_set, args.model_name)
    out_dir = os.path.dirname(output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    report.save(output)
    print(f"Report saved to {output}")
    report.print_summary()


# ── validate ───────────────────────────────────────────────────────────────

def _validate(args: argparse.Namespace) -> None:
    if not os.path.exists(args.task_set):
        print(f"Error: file not found — {args.task_set}")
        return

    task_set = TaskSet.from_jsonl(args.task_set)
    report = TaskValidator().validate_set(task_set)
    print_report(report)

    if args.output:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {args.output}")


# ── entry point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="algebraid",
        description="ALGEBRAID — procedurally generated benchmark for compositional generalization.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # generate
    p = sub.add_parser("generate", help="Generate a new task set.")
    p.add_argument("-o", "--output", default=None,
                   help="Output JSONL path (default: ./data/algebraid_s{seed}_{date}.jsonl).")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    p.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 4],
                   help="Composition depths (default: 1 2 3 4).")
    p.add_argument("--tasks-per-depth", type=int, default=50,
                   help="Tasks per depth/family (default: 50).")
    p.add_argument("--families", nargs="+", default=["intra", "inter", "field", "rule"],
                   help="Task families (default: intra inter field rule).")
    p.add_argument("--no-dims", action="store_true",
                   help="Exclude Hupkes compositionality dimensions.")
    p.add_argument("--skip-validation", action="store_true",
                   help="Skip post-generation validation.")
    p.set_defaults(func=_generate)

    # run
    p = sub.add_parser("run", help="Run tasks against a model.")
    p.add_argument("task_set", help="Path to the task set JSONL.")
    p.add_argument("-o", "--output", default=None,
                   help="Output predictions JSON (default: ./results/preds_{model}_{taskset}.json).")
    p.add_argument("-a", "--adapter", default="openai",
                   help="Adapter: openai | anthropic | huggingface | custom_http.")
    p.add_argument("-m", "--model", default="gpt-4.1-nano",
                   help="Model identifier (default: gpt-4.1-nano).")
    p.add_argument("-t", "--temperature", type=float, default=0.0,
                   help="Sampling temperature (default: 0.0).")
    p.add_argument("--max-tokens", type=int, default=512,
                   help="Max response tokens (default: 512).")
    p.add_argument("--delay", type=float, default=0.5,
                   help="Delay between API calls in seconds (default: 0.5).")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="Suppress per-task progress output.")
    p.set_defaults(func=_run)

    # evaluate
    p = sub.add_parser("evaluate", help="Score predictions against a task set.")
    p.add_argument("task_set", help="Path to the task set JSONL.")
    p.add_argument("predictions", help="Path to the predictions JSON.")
    p.add_argument("-o", "--output", default=None,
                   help="Output report JSON (default: ./results/report_{model}_{taskset}.json).")
    p.add_argument("--model-name", default="unknown",
                   help="Model display name for the report.")
    p.add_argument("--strict", action="store_true",
                   help="Enable strict answer matching.")
    p.set_defaults(func=_evaluate)

    # validate
    p = sub.add_parser("validate", help="Validate a task set for quality issues.")
    p.add_argument("task_set", help="Path to the task set JSONL.")
    p.add_argument("-o", "--output", default=None,
                   help="Output validation report JSON (default: stdout only).")
    p.set_defaults(func=_validate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
