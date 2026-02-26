"""
Command-line interface for ALGEBRAID.

Commands
--------
generate   Create a new task set.
run        Execute a task set against a language model (produces predictions JSON).
evaluate   Score model predictions against a task set.
pipeline   Single command: run + evaluate + prove + analyze, all in one run folder.
validate   Check a task set for structural or semantic issues.
prove      Independently re-derive every answer to verify algebraic correctness.
analyze    Compute scaling laws, phase transitions, and error taxonomy.
split      Partition a task set into train/test splits for generalisation studies.
"""

import argparse
import hashlib
import os
import json
import re
import time
from datetime import datetime, date
from pathlib import Path

from .generator import AlgebraidGenerator
from .evaluator import AlgebraidEvaluator, EvalReport
from .task_model import TaskSet
from .adapters import get_adapter
from .tasks.validator import TaskValidator, print_report
from .proof import verify_set, print_proof_report
from .analysis import run_analysis, print_analysis
from .splits import (
    split_by_depth, split_by_commutativity, split_by_structure,
    split_by_family, split_summary,
)


# ── naming helpers ──────────────────────────────────────────────────────────

def _config_hash(depths: list, tasks_per_depth: int, families: list) -> str:
    """Return a short 6-char hex hash of the generation configuration."""
    raw = f"{sorted(depths)}-{tasks_per_depth}-{sorted(families)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:6]


def _default_task_path(seed: int, depths: list, tasks_per_depth: int, families: list) -> str:
    """Default task-set path: ``./data/algebraid_s{seed}_{YYYYMMDD}_{config_hash}.jsonl``"""
    h = _config_hash(depths, tasks_per_depth, families)
    return f"./data/algebraid_s{seed}_{date.today():%Y%m%d}_{h}.jsonl"


def _sanitize(name: str) -> str:
    """Turn an arbitrary string into a filesystem-safe slug."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()


def _stem(path: str) -> str:
    """Extract the filename stem (no directory, no extension)."""
    return Path(path).stem


def _timestamp() -> str:
    """Return a compact timestamp: ``YYYYMMDD_HHMMSS``."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _run_id(model: str, task_set_path: str) -> str:
    """
    Build a unique, human-readable run identifier.

    Pattern: ``{YYYYMMDD_HHMMSS}_{sanitized_model}_{taskset_stem}``

    Example: ``20260225_143012_gpt_4_1_nano_algebraid_s42_20260225_e50604``

    The timestamp guarantees uniqueness; the model + task-set names make
    the folder self-describing without opening any file.
    """
    return f"{_timestamp()}_{_sanitize(model)}_{_stem(task_set_path)}"


def _default_predictions_path(task_set_path: str, model: str) -> str:
    """``./results/preds_{model}_{taskset}_{timestamp}.json``"""
    return f"./results/preds_{_sanitize(model)}_{_stem(task_set_path)}_{_timestamp()}.json"


def _default_report_path(task_set_path: str, model_name: str) -> str:
    """``./results/report_{model}_{taskset}_{timestamp}.json``"""
    return f"./results/report_{_sanitize(model_name)}_{_stem(task_set_path)}_{_timestamp()}.json"


# ── generate ────────────────────────────────────────────────────────────────

def _generate(args: argparse.Namespace) -> None:
    output = args.output or _default_task_path(
        args.seed, args.depths, args.tasks_per_depth, args.families
    )
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


# ── run ─────────────────────────────────────────────────────────────────────

def _run(args: argparse.Namespace) -> None:
    if not os.path.exists(args.task_set):
        print(f"Error: task set not found: {args.task_set}")
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


# ── evaluate ────────────────────────────────────────────────────────────────

def _evaluate(args: argparse.Namespace) -> None:
    for path, label in [(args.task_set, "Task set"), (args.predictions, "Predictions")]:
        if not os.path.exists(path):
            print(f"Error: {label} not found: {path}")
            return

    task_set = TaskSet.from_jsonl(args.task_set)
    with open(args.predictions) as f:
        predictions = json.load(f)

    ts = _timestamp()
    rid = f"{ts}_{_sanitize(args.model_name)}_{_stem(args.task_set)}"
    evaluator = AlgebraidEvaluator(strict=args.strict)
    report = evaluator.evaluate(
        task_set, predictions, model_name=args.model_name,
        run_id=rid, timestamp=ts,
    )

    output = args.output or _default_report_path(args.task_set, args.model_name)
    out_dir = os.path.dirname(output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    report.save(output)
    print(f"Report saved to {output}")
    report.print_summary()


# ── pipeline ────────────────────────────────────────────────────────────────

def _pipeline(args: argparse.Namespace) -> None:
    """
    Single-command evaluation pipeline: run -> evaluate -> prove -> analyze.

    Creates a uniquely named run folder containing all artifacts:

        {output_dir}/{run_id}/
            manifest.json        run metadata and artifact paths
            predictions.json     raw model responses
            eval_report.json     accuracy + complexity metrics (with per-task results)
            proof_report.json    algebraic correctness verification
            analysis.json        scaling laws, phase transitions, error taxonomy
    """
    if not os.path.exists(args.task_set):
        print(f"Error: task set not found: {args.task_set}")
        return

    # Build unique run ID and folder
    rid = _run_id(args.model, args.task_set)
    base_dir = args.output_dir or "./runs"
    run_dir = os.path.join(base_dir, rid)
    os.makedirs(run_dir, exist_ok=True)
    ts = rid[:15]  # YYYYMMDD_HHMMSS portion

    print(f"Run folder: {run_dir}")

    task_set = TaskSet.from_jsonl(args.task_set)
    total_steps = 2 + (0 if args.skip_prove else 1) + (0 if args.skip_analyze else 1)
    step = 1

    # ── Step 1: Run model ───────────────────────────────────────────────────
    print(f"\n[{step}/{total_steps}] Running {len(task_set)} tasks on {args.model} ...")
    step += 1

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
    elapsed = time.time() - t0
    print(f"      Completed in {elapsed:.1f}s")

    preds_path = os.path.join(run_dir, "predictions.json")
    with open(preds_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"      Saved -> predictions.json")

    # ── Step 2: Evaluate ────────────────────────────────────────────────────
    print(f"\n[{step}/{total_steps}] Evaluating predictions ...")
    step += 1

    evaluator = AlgebraidEvaluator(strict=args.strict)
    report = evaluator.evaluate(
        task_set, predictions,
        model_name=args.model,
        run_id=rid,
        timestamp=ts,
    )
    if report.missing_predictions:
        print(f"      WARNING: {report.missing_predictions} task(s) had no prediction (scored wrong)")

    report_path = os.path.join(run_dir, "eval_report.json")
    report.save(report_path)
    print(f"      Saved -> eval_report.json")
    report.print_summary()

    # ── Step 3: Prove ───────────────────────────────────────────────────────
    proof_path = None
    if not args.skip_prove:
        print(f"\n[{step}/{total_steps}] Running algebraic proof verification ...")
        step += 1

        proof = verify_set(task_set)
        print_proof_report(proof)

        proof_path = os.path.join(run_dir, "proof_report.json")
        with open(proof_path, "w") as f:
            json.dump(proof, f, indent=2)
        print(f"      Saved -> proof_report.json")
    else:
        print(f"\n[{step}/{total_steps}] Proof verification skipped (--skip-prove).")
        step += 1

    # ── Step 4: Analyze ─────────────────────────────────────────────────────
    analysis_path = None
    if not args.skip_analyze:
        print(f"\n[{step}/{total_steps}] Running error analysis ...")

        analysis = run_analysis(report)
        print_analysis(analysis)

        analysis_path = os.path.join(run_dir, "analysis.json")
        saveable = {k: v for k, v in analysis.items() if k != "results"}
        with open(analysis_path, "w") as f:
            json.dump(saveable, f, indent=2)
        print(f"      Saved -> analysis.json")
    else:
        print(f"\n[{step}/{total_steps}] Error analysis skipped (--skip-analyze).")

    # ── Manifest ────────────────────────────────────────────────────────────
    manifest = {
        "run_id": rid,
        "timestamp": ts,
        "task_set": os.path.abspath(args.task_set),
        "task_count": len(task_set),
        "model": args.model,
        "adapter": args.adapter,
        "temperature": args.temperature,
        "strict_grading": args.strict,
        "overall_accuracy": round(report.accuracy_overall, 4),
        "total_correct": report.total_correct,
        "total_tasks": report.total_tasks,
        "missing_predictions": report.missing_predictions,
        "artifacts": {
            "predictions": "predictions.json",
            "eval_report": "eval_report.json",
            **({"proof_report": "proof_report.json"} if proof_path else {}),
            **({"analysis": "analysis.json"} if analysis_path else {}),
        },
    }
    manifest_path = os.path.join(run_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Pipeline complete")
    print(f"  Run folder : {run_dir}")
    print(f"  Accuracy   : {report.accuracy_overall:.1%} ({report.total_correct}/{report.total_tasks})")
    print(f"{'='*60}\n")


# ── validate ────────────────────────────────────────────────────────────────

def _validate(args: argparse.Namespace) -> None:
    if not os.path.exists(args.task_set):
        print(f"Error: task set not found: {args.task_set}")
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


# ── prove ────────────────────────────────────────────────────────────────────

def _prove(args: argparse.Namespace) -> None:
    if not os.path.exists(args.task_set):
        print(f"Error: task set not found: {args.task_set}")
        return

    task_set = TaskSet.from_jsonl(args.task_set)
    print(f"Verifying {len(task_set)} tasks ...")
    report = verify_set(task_set)
    print_proof_report(report)

    if args.output:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Proof report saved to {args.output}")


# ── analyze ──────────────────────────────────────────────────────────────────

def _analyze(args: argparse.Namespace) -> None:
    if not os.path.exists(args.report):
        print(f"Error: report not found: {args.report}")
        return

    with open(args.report) as f:
        raw = json.load(f)

    # Full reconstruction including per-task results (needed for error taxonomy)
    report = EvalReport.from_dict(raw)

    if not report.results:
        print("NOTE: Report contains no per-task results; error taxonomy will be empty.")
        print("      Re-run evaluation to get a report with full results.")

    analysis = run_analysis(report)
    print_analysis(analysis)

    if args.output:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        saveable = {k: v for k, v in analysis.items() if k != "results"}
        with open(args.output, "w") as f:
            json.dump(saveable, f, indent=2)
        print(f"Analysis saved to {args.output}")


# ── split ────────────────────────────────────────────────────────────────────

def _split(args: argparse.Namespace) -> None:
    if not os.path.exists(args.task_set):
        print(f"Error: task set not found: {args.task_set}")
        return

    task_set = TaskSet.from_jsonl(args.task_set)
    stem = _stem(args.task_set)
    out_dir = args.output_dir or "./data"
    os.makedirs(out_dir, exist_ok=True)

    mode = args.mode

    if mode == "depth":
        train, test = split_by_depth(task_set, args.train_max_depth, args.test_min_depth)
    elif mode == "commutativity":
        train, test = split_by_commutativity(task_set)
    elif mode == "structure":
        if not args.train_prefixes or not args.test_prefixes:
            print("Error: --train-prefixes and --test-prefixes required for 'structure' mode.")
            return
        train, test = split_by_structure(task_set, args.train_prefixes, args.test_prefixes)
    elif mode == "family":
        if not args.train_families or not args.test_families:
            print("Error: --train-families and --test-families required for 'family' mode.")
            return
        train, test = split_by_family(task_set, args.train_families, args.test_families)
    else:
        print(f"Unknown split mode: {mode}")
        return

    print(split_summary(train, test))

    train_path = os.path.join(out_dir, f"{stem}_{mode}_train.jsonl")
    test_path = os.path.join(out_dir, f"{stem}_{mode}_test.jsonl")
    train.to_jsonl(train_path)
    test.to_jsonl(test_path)
    print(f"Train split -> {train_path}")
    print(f"Test  split -> {test_path}")


# ── entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="algebraid",
        description="ALGEBRAID: compositional algebraic reasoning benchmark.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # generate
    p = sub.add_parser("generate", help="Generate a new task set.")
    p.add_argument("-o", "--output", default=None,
                   help="Output JSONL path (default: ./data/algebraid_s{seed}_{date}_{hash}.jsonl).")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    p.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 4],
                   help="Composition depths (default: 1 2 3 4).")
    p.add_argument("--tasks-per-depth", type=int, default=50,
                   help="Tasks per depth/family (default: 50).")
    p.add_argument(
        "--families", nargs="+",
        default=["intra", "inter", "field", "rule", "conceptual", "adversarial", "intermediate"],
        choices=["intra", "inter", "field", "rule", "conceptual", "adversarial", "intermediate"],
        help=(
            "Task families to generate (default: all seven). "
            "intra=chain composition, inter=direct product, field=GF(p) arithmetic, "
            "rule=pattern induction, conceptual=structural queries, "
            "adversarial=reversible chain traps, intermediate=partial chain queries."
        ),
    )
    p.add_argument("--no-dims", action="store_true",
                   help="Exclude Hupkes compositionality dimensions.")
    p.add_argument("--skip-validation", action="store_true",
                   help="Skip post-generation validation.")
    p.set_defaults(func=_generate)

    # run
    p = sub.add_parser("run", help="Run a task set against a language model.")
    p.add_argument("task_set", help="Path to the task set JSONL.")
    p.add_argument("-o", "--output", default=None,
                   help="Output predictions JSON (default: ./results/preds_{model}_{taskset}_{ts}.json).")
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
                   help="Output report JSON (default: ./results/report_{model}_{taskset}_{ts}.json).")
    p.add_argument("--model-name", default="unknown",
                   help="Model display name for the report.")
    p.add_argument("--strict", action="store_true",
                   help="Enable strict answer matching.")
    p.set_defaults(func=_evaluate)

    # pipeline
    p = sub.add_parser(
        "pipeline",
        help=(
            "Full evaluation pipeline in one command: "
            "run model -> evaluate -> prove -> analyze. "
            "All artifacts are saved to a uniquely named run folder."
        ),
    )
    p.add_argument("task_set", help="Path to the task set JSONL.")
    p.add_argument("-m", "--model", default="gpt-4.1-nano",
                   help="Model identifier (default: gpt-4.1-nano).")
    p.add_argument("-a", "--adapter", default="openai",
                   help="Adapter: openai | anthropic | huggingface | custom_http.")
    p.add_argument("-t", "--temperature", type=float, default=0.0,
                   help="Sampling temperature (default: 0.0).")
    p.add_argument("--max-tokens", type=int, default=512,
                   help="Max response tokens (default: 512).")
    p.add_argument("--delay", type=float, default=0.5,
                   help="Delay between API calls in seconds (default: 0.5).")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="Suppress per-task progress output.")
    p.add_argument("--strict", action="store_true",
                   help="Enable strict answer matching.")
    p.add_argument("--skip-prove", action="store_true",
                   help="Skip algebraic proof verification step.")
    p.add_argument("--skip-analyze", action="store_true",
                   help="Skip error scaling / phase transition analysis step.")
    p.add_argument("-o", "--output-dir", default=None,
                   help="Base directory for run folders (default: ./runs).")
    p.set_defaults(func=_pipeline)

    # validate
    p = sub.add_parser("validate", help="Validate a task set for quality issues.")
    p.add_argument("task_set", help="Path to the task set JSONL.")
    p.add_argument("-o", "--output", default=None,
                   help="Output validation report JSON (default: stdout only).")
    p.set_defaults(func=_validate)

    # prove
    p = sub.add_parser(
        "prove",
        help="Independently re-derive every answer to verify algebraic correctness.",
    )
    p.add_argument("task_set", help="Path to the task set JSONL.")
    p.add_argument("-o", "--output", default=None,
                   help="Output proof report JSON (default: stdout only).")
    p.set_defaults(func=_prove)

    # analyze
    p = sub.add_parser(
        "analyze",
        help="Compute error scaling laws, phase transitions, and error taxonomy from an eval report.",
    )
    p.add_argument("report", help="Path to an evaluation report JSON (from 'evaluate' or 'pipeline').")
    p.add_argument("-o", "--output", default=None,
                   help="Output analysis JSON (default: stdout only).")
    p.set_defaults(func=_analyze)

    # split
    p = sub.add_parser(
        "split",
        help="Partition a task set into train/test splits for generalisation studies.",
    )
    p.add_argument("task_set", help="Path to the task set JSONL.")
    p.add_argument(
        "--mode", required=True,
        choices=["depth", "commutativity", "structure", "family"],
        help=(
            "Split strategy: "
            "depth=by composition depth, "
            "commutativity=abelian vs non-abelian, "
            "structure=by structure name prefix, "
            "family=by task family."
        ),
    )
    p.add_argument("--train-max-depth", type=int, default=2,
                   help="[depth mode] Max depth in train set (default: 2).")
    p.add_argument("--test-min-depth", type=int, default=None,
                   help="[depth mode] Min depth in test set (default: train-max-depth + 2).")
    p.add_argument("--train-prefixes", nargs="+", default=None,
                   help="[structure mode] Structure name prefixes for training, e.g. Z_ GF(.")
    p.add_argument("--test-prefixes", nargs="+", default=None,
                   help="[structure mode] Structure name prefixes for testing.")
    p.add_argument("--train-families", nargs="+", default=None,
                   help="[family mode] Task families for training, e.g. intra inter.")
    p.add_argument("--test-families", nargs="+", default=None,
                   help="[family mode] Task families for testing.")
    p.add_argument("--output-dir", default=None,
                   help="Directory for split JSONL files (default: ./data).")
    p.set_defaults(func=_split)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
