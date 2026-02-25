"""
ALGEBRAID Command-Line Interface.

Provides four main commands:
- `algebraid generate`: Generate a new task set.
- `algebraid run`: Run a task set against a model.
- `algebraid evaluate`: Evaluate model predictions.
- `algebraid validate`: Validate a task set for quality issues.
"""

import argparse
import os
import json
import time

from .generator import AlgebraidGenerator
from .evaluator import AlgebraidEvaluator
from .task_model import TaskSet
from .adapters import get_adapter
from .tasks.validator import TaskValidator, print_validation_report


def _generate_handler(args):
    """Handle the `generate` subcommand."""
    print(f"Generating task set with seed {args.seed}...")
    gen = AlgebraidGenerator(seed=args.seed)
    task_set = gen.generate(
        depths=args.depths,
        tasks_per_depth=args.tasks_per_depth,
        families=args.families,
        include_dimensions=not args.no_dims,
    )
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    task_set.to_jsonl(args.output)
    print(f"Task set with {len(task_set)} tasks saved to {args.output}")
    print(task_set.summary())

    # Auto-validate unless --skip-validation is set
    if not args.skip_validation:
        print("Running post-generation validation...")
        validator = TaskValidator()
        report = validator.validate_taskset(task_set)
        print_validation_report(report)
        if report["failed"] > 0:
            print(f"WARNING: {report['failed']} tasks failed validation. Review errors above.")


def _run_handler(args):
    """Handle the `run` subcommand."""
    if not os.path.exists(args.task_set):
        print(f"Error: Task set file not found at {args.task_set}")
        return

    print(f"Loading tasks from {args.task_set}...")
    task_set = TaskSet.from_jsonl(args.task_set)
    print(f"Running {len(task_set)} tasks on model: {args.model}")

    Adapter = get_adapter(args.adapter)
    adapter = Adapter(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        delay=args.delay,
        verbose=not args.quiet,
    )

    start_time = time.time()
    predictions = adapter.run_tasks(task_set)
    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.1f}s")

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"Predictions saved to {args.output}")


def _evaluate_handler(args):
    """Handle the `evaluate` subcommand."""
    if not os.path.exists(args.task_set):
        print(f"Error: Task set file not found at {args.task_set}")
        return
    if not os.path.exists(args.predictions):
        print(f"Error: Predictions file not found at {args.predictions}")
        return

    print("Loading tasks and predictions...")
    task_set = TaskSet.from_jsonl(args.task_set)
    with open(args.predictions) as f:
        predictions = json.load(f)

    print("Evaluating...")
    evaluator = AlgebraidEvaluator(strict=args.strict)
    report = evaluator.evaluate(task_set, predictions, model_name=args.model_name)

    if args.output:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        report.save(args.output)
        print(f"Evaluation report saved to {args.output}")

    report.print_summary()


def _validate_handler(args):
    """Handle the `validate` subcommand."""
    if not os.path.exists(args.task_set):
        print(f"Error: Task set file not found at {args.task_set}")
        return

    print(f"Validating task set: {args.task_set}")
    task_set = TaskSet.from_jsonl(args.task_set)
    validator = TaskValidator()
    report = validator.validate_taskset(task_set)
    print_validation_report(report)

    if args.output:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Validation report saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(
        prog="algebraid",
        description="ALGEBRAID: A procedurally generated benchmark for compositional generalization.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── generate ──────────────────────────────────────────────────────────────
    p_gen = subparsers.add_parser("generate", help="Generate a new task set.")
    p_gen.add_argument(
        "-o", "--output", default="./data/task_set.jsonl",
        help="Output path for task set JSONL file. (default: ./data/task_set.jsonl)"
    )
    p_gen.add_argument("--seed", type=int, default=42, help="Random seed. (default: 42)")
    p_gen.add_argument(
        "--depths", type=int, nargs="+", default=[1, 2, 3, 4],
        help="Composition depths to generate. (default: 1 2 3 4)"
    )
    p_gen.add_argument(
        "--tasks-per-depth", type=int, default=50,
        help="Tasks per depth/family combination. (default: 50)"
    )
    p_gen.add_argument(
        "--families", nargs="+", default=["intra", "inter", "field", "rule"],
        help="Task families: intra inter field rule. (default: all)"
    )
    p_gen.add_argument(
        "--no-dims", action="store_true",
        help="Exclude Hupkes compositionality dimensions."
    )
    p_gen.add_argument(
        "--skip-validation", action="store_true",
        help="Skip post-generation validation."
    )
    p_gen.set_defaults(func=_generate_handler)

    # ── run ───────────────────────────────────────────────────────────────────
    p_run = subparsers.add_parser("run", help="Run a task set against a model.")
    p_run.add_argument("task_set", help="Path to the task set JSONL file.")
    p_run.add_argument(
        "-o", "--output", default="./results/predictions.json",
        help="Output path for predictions JSON. (default: ./results/predictions.json)"
    )
    p_run.add_argument(
        "-a", "--adapter", default="openai",
        help="Adapter to use: openai | anthropic | huggingface | custom_http. (default: openai)"
    )
    p_run.add_argument(
        "-m", "--model", default="gpt-4.1-nano",
        help="Model name to query. (default: gpt-4.1-nano)"
    )
    p_run.add_argument(
        "-t", "--temperature", type=float, default=0.0,
        help="Sampling temperature. (default: 0.0)"
    )
    p_run.add_argument(
        "--max-tokens", type=int, default=512,
        help="Max tokens per response. (default: 512)"
    )
    p_run.add_argument(
        "--delay", type=float, default=0.5,
        help="Delay between API calls in seconds. (default: 0.5)"
    )
    p_run.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output.")
    p_run.set_defaults(func=_run_handler)

    # ── evaluate ──────────────────────────────────────────────────────────────
    p_eval = subparsers.add_parser("evaluate", help="Evaluate model predictions against a task set.")
    p_eval.add_argument("task_set", help="Path to the task set JSONL file.")
    p_eval.add_argument("predictions", help="Path to the predictions JSON file.")
    p_eval.add_argument(
        "-o", "--output", default="./results/report.json",
        help="Output path for the evaluation report JSON. (default: ./results/report.json)"
    )
    p_eval.add_argument(
        "--model-name", default="unknown_model",
        help="Display name of the model for the report."
    )
    p_eval.add_argument("--strict", action="store_true", help="Enable strict answer matching.")
    p_eval.set_defaults(func=_evaluate_handler)

    # ── validate ──────────────────────────────────────────────────────────────
    p_val = subparsers.add_parser("validate", help="Validate a task set for quality issues.")
    p_val.add_argument("task_set", help="Path to the task set JSONL file.")
    p_val.add_argument(
        "-o", "--output", default=None,
        help="Output path for validation report JSON. (default: print only)"
    )
    p_val.set_defaults(func=_validate_handler)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
