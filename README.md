# ALGEBRAID

**A procedurally generated benchmark for compositional reasoning in language models, grounded in formal algebraic structures.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

ALGEBRAID generates an unlimited, non-memorisable supply of reasoning tasks built on five algebraic structures and seven task families. Every answer is formally guaranteed correct — either by algebraic construction or by an independent proof verifier that re-derives each answer from scratch. Because tasks are generated from a seed, any experiment can be reproduced exactly.

## Key Properties

| Property | Detail |
|---|---|
| **Guaranteed correct answers** | All answers are derived by computation, not labelling |
| **Formal verification** | Independent re-derivation checks every solution trace |
| **No memorisation** | Unlimited procedural variation; no fixed test set |
| **Reproducible** | Seed-based generation; identical seed → identical task set |
| **Multi-dimensional** | 7 task families × 7 Hupkes dimensions × 4 complexity metrics |

## Install

```bash
pip install -e .

# Optional: Anthropic adapter (to evaluate Claude models)
pip install -e ".[anthropic]"
```

Set your API key before running:

```bash
export OPENAI_API_KEY=sk-...        # for openai / custom_http adapters
export ANTHROPIC_API_KEY=sk-ant-... # for anthropic adapter
```

## Quick Start

```bash
# 1. Generate a task set (validated automatically)
algebraid generate --seed 42
# -> ./data/algebraid_s42_20260225_e50604.jsonl

# 2. Run the full pipeline in one command (run + evaluate + prove + analyze)
algebraid pipeline ./data/algebraid_s42_20260225_e50604.jsonl \
    --model gpt-4.1-nano --adapter openai
# -> ./runs/20260225_143012_gpt_4_1_nano_algebraid_s42_.../
```

The pipeline creates a self-contained, uniquely-named run folder — see [Pipeline Output](#pipeline-output).

---

## Task Families

ALGEBRAID generates seven task families, controlled via `--families`:

| Family key | Description | Example |
|---|---|---|
| `intra` | Composition chain within a single structure | Apply inverse, then multiply by 3, then square in Z₁₂ |
| `inter` | Component-wise operation across a direct product | Compute (Z₇ × S₃) element-wise |
| `field` | Expression tree in a finite field GF(p) | Evaluate `(3 * x + 5) mod 7` |
| `rule` | Identify the pattern and give the next element | Find the rule in a Z₅ sequence |
| `conceptual` | Structural property query (no computation chain) | What is the order of element 4 in Z₁₂? |
| `adversarial` | Reversible chain designed to trigger specific errors | Apply inverse twice — answer is the starting value |
| `intermediate` | Report the value after step k of an n-step chain | What is the value after step 2 of a depth-4 chain? |

All families are included by default. The conceptual, adversarial, and intermediate families expose distinct failure modes invisible to standard chain tasks.

### Adversarial Sub-types

| Sub-type | Trap | Wrong answer |
|---|---|---|
| `double_inverse` | Two inverses cancel — answer = start | Applying only one inverse |
| `self_cancelling` | Operations sum to zero mod n | Intermediate value after first op |
| `identity_bait` | Result is the identity element | Starting value |
| `commutativity_trap` | Non-abelian structure; order matters | Swapped operand result |

---

## Algebraic Structures

| Structure | Notation | Non-abelian | Order |
|---|---|---|---|
| Cyclic group | Z_n | No | n |
| Symmetric group | S_n (n = 3–5) | Yes (n ≥ 3) | n! |
| Dihedral group | D_n (n = 3–8) | Yes (n > 2) | 2n |
| Finite field | GF(p) | No | p |
| Quaternion group | Q_8 | Yes | 8 |

---

## Semantic Skins

Every task can be expressed through a narrative skin that translates abstract algebra into a coherent real-world context:

| Skin | Structure | Domain |
|---|---|---|
| Clock Arithmetic | Z_n | Positions on an n-hour clock |
| Musical Intervals | Z_n | Tones on a chromatic scale |
| Robot Steps | Z_n | Stops on a circular track |
| Color Wheel | Z_n | Hues on a colour wheel |
| Deck of Cards | S_n | Shuffles of n cards |
| Seating Arrangements | S_n | Rearrangements of n people |
| Polygon Symmetries | D_n | Symmetries of a regular polygon |
| Tile Flips and Rotations | D_n | Orientations of a tile |
| Secret Codes | GF(p) | Symbols in a code system |
| Modular Arithmetic | GF(p) | Residues modulo a prime |
| Quaternion Algebra | Q_8 | Quaternion unit arithmetic |
| Quaternion Rotations | Q_8 | 3-D rotation states |

---

## CLI Reference

### `algebraid generate`

Create a task set and validate it immediately.

```bash
algebraid generate --seed 42 --depths 1 2 3 4 --tasks-per-depth 50
```

| Argument | Default | Description |
|---|---|---|
| `-o`, `--output` | `./data/algebraid_s{seed}_{date}_{hash}.jsonl` | Output path |
| `--seed` | `42` | Random seed |
| `--depths` | `1 2 3 4` | Composition depths to generate |
| `--tasks-per-depth` | `50` | Tasks per depth/family combination |
| `--families` | all seven | Space-separated list of families to include |
| `--no-dims` | | Exclude Hupkes compositionality dimension variants |
| `--skip-validation` | | Skip post-generation schema validation |

### `algebraid pipeline`

End-to-end evaluation in a single command. Runs the model, scores predictions, verifies algebraic correctness, and computes error analysis — all saved to a uniquely named folder.

```bash
algebraid pipeline ./data/tasks.jsonl --model claude-sonnet-4-6 --adapter anthropic
```

| Argument | Default | Description |
|---|---|---|
| `task_set` | | Path to task set JSONL |
| `-m`, `--model` | `gpt-4.1-nano` | Model identifier |
| `-a`, `--adapter` | `openai` | `openai` \| `anthropic` \| `custom_http` \| `huggingface` |
| `-t`, `--temperature` | `0.0` | Sampling temperature |
| `--max-tokens` | `512` | Max response tokens |
| `--delay` | `0.5` | Seconds between API calls |
| `--strict` | | Strict answer matching |
| `--skip-prove` | | Skip algebraic proof verification |
| `--skip-analyze` | | Skip error scaling / phase analysis |
| `-o`, `--output-dir` | `./runs` | Base directory for run folders |

### `algebraid run`

Run a task set against a model (produces predictions JSON only).

```bash
algebraid run ./data/tasks.jsonl -m gpt-4.1-nano -q
```

### `algebraid evaluate`

Score an existing predictions file against a task set.

```bash
algebraid evaluate ./data/tasks.jsonl ./results/preds_*.json --model-name gpt-4.1-nano
```

### `algebraid validate`

Check a task set for schema errors, prompt quality issues, and answer inconsistencies.

```bash
algebraid validate ./data/tasks.jsonl
```

### `algebraid prove`

Independently re-derive every answer from its solution trace using algebraic primitives — no reference to the generator's computation path.

```bash
algebraid prove ./data/tasks.jsonl
```

### `algebraid analyze`

Compute error scaling laws, phase transitions, and mechanistic error taxonomy from an evaluation report.

```bash
algebraid analyze ./runs/20260225_143012_.../eval_report.json
```

### `algebraid split`

Partition a task set into train/test splits for compositional generalisation studies.

```bash
algebraid split ./data/tasks.jsonl --mode depth --train-max-depth 2
algebraid split ./data/tasks.jsonl --mode commutativity
algebraid split ./data/tasks.jsonl --mode structure --train-prefixes Z_ GF( --test-prefixes S_ D_
algebraid split ./data/tasks.jsonl --mode family --train-families intra inter --test-families adversarial
```

---

## Pipeline Output

The `pipeline` command creates a uniquely identifiable folder per run:

```
./runs/20260225_143012_gpt_4_1_nano_algebraid_s42_20260225_e50604/
    manifest.json       model, task set, accuracy summary, artifact paths
    predictions.json    raw model responses keyed by task_id
    eval_report.json    accuracy by depth / family / dimension + complexity metrics
    proof_report.json   algebraic verification: proven / skipped / failed counts
    analysis.json       power-law fit, phase transition, error taxonomy, hallucination onset
```

The folder name encodes the timestamp, model, and task set — making runs sortable and self-describing without opening any file.

The `eval_report.json` includes per-task results (response, ground truth, correctness) capped at 512 characters per response. This enables `algebraid analyze` to produce a complete error taxonomy from a saved report without re-running the model.

---

## Output Naming Convention

All default output paths are self-describing:

| Artifact | Pattern |
|---|---|
| Task set | `./data/algebraid_s{seed}_{YYYYMMDD}_{config_hash}.jsonl` |
| Predictions | `./results/preds_{model}_{taskset}_{timestamp}.json` |
| Eval report | `./results/report_{model}_{taskset}_{timestamp}.json` |
| Run folder | `./runs/{YYYYMMDD_HHMMSS}_{model}_{taskset}/` |

The 6-character `config_hash` in task set names is derived from `(depths, tasks_per_depth, families)`, so different generation parameters never collide even with the same seed and date.

---

## Python API

```python
from algebraid import (
    AlgebraidGenerator, AlgebraidEvaluator, EvalReport,
    TaskValidator, verify_set, run_analysis,
    split_by_depth, split_by_commutativity,
)

# Generate
gen = AlgebraidGenerator(seed=42)
task_set = gen.generate(
    depths=[1, 2, 3, 4],
    tasks_per_depth=20,
    families=["intra", "inter", "conceptual", "adversarial"],
)

# Validate
val_report = TaskValidator().validate_set(task_set)
print(f"{val_report['passed']}/{val_report['total']} passed")

# Prove (re-derive all answers independently)
proof = verify_set(task_set)
print(f"Proof rate: {proof['proof_rate']}%")

# Evaluate (given a predictions dict)
predictions = {"AG-abc123": "5", ...}
evaluator = AlgebraidEvaluator()
report = evaluator.evaluate(task_set, predictions, model_name="my-model")
report.print_summary()

# Analyze
analysis = run_analysis(report)

# Generalisation splits
train, test = split_by_depth(task_set, train_max_depth=2)
train, test = split_by_commutativity(task_set)

# Load a saved report and re-analyze
with open("eval_report.json") as f:
    import json
    report = EvalReport.from_dict(json.load(f))
analysis = run_analysis(report)
```

---

## Algebraic Complexity Metrics

Every task carries four algebraic complexity scores, computed independently of the model:

| Metric | Symbol | Measures |
|---|---|---|
| Algebraic Entropy | H_alg | log₂(\|G\|) × depth — information needed to specify a result |
| Commutativity Distance | D_comm | Fraction of non-commuting element pairs in the chain |
| Orbit Complexity | O_c | Distinct intermediate values / \|G\| — breadth of traversal |
| Structural Interference | I_s | Coprimality-based interference for direct products |

Two extended metrics apply to the new task families:

- **`compute_conceptual_depth(task)`** — difficulty of a conceptual query (0.1 for identity lookup, 0.7 for element-order computation)
- **`compute_adversarial_strength(task)`** — strength of the adversarial trap (0.3 for self-cancellation, 0.8 for commutativity trap)

---

## Compositional Dimensions

Following Hupkes et al. (2020), tasks are tagged with one of seven dimensions:

| Dimension | Tests |
|---|---|
| `general` | Standard task (no specific dimension stress) |
| `systematicity` | Unseen combination of seen components |
| `substitutivity` | Synonym substitution invariance |
| `productivity` | Longer chain than seen during training |
| `overgeneralization` | Resistance to applying rules where they don't hold |
| `adversarial` | Resistance to specific reasoning shortcuts |
| `intermediate_state` | Access to intermediate computation steps |

---

## Project Layout

```
src/algebraid/
    primitives/         Z_n, S_n, D_n, GF(p), Q_8 + abstract base
    composers/          FunctionComposition, DirectProduct
    tasks/              Verbalization, answer verification, schema validation
    generator.py        Procedural task generation (7 families)
    evaluator.py        EvalReport, AlgebraidEvaluator
    proof.py            Independent algebraic verifier
    analysis.py         Scaling laws, phase transitions, error taxonomy
    complexity.py       4 algebraic complexity metrics + extended metrics
    splits.py           4 generalisation split strategies
    skins.py            12 semantic narrative skins
    adapters.py         OpenAI, Anthropic, custom HTTP model adapters
    cli.py              8-command CLI (generate, run, evaluate, pipeline, ...)
tests/
    test_primitives.py  Group axiom verification for all 5 structures
    test_composers.py   Composition and direct product tests
    test_generator.py   Determinism, schema, and grading tests (7 families)
    test_complexity.py  Algebraic complexity metric tests
    test_task_model.py  Task/TaskSet serialization round-trip tests
    test_verbalizer.py  Prompt generation and skin integration tests
    test_verifier.py    Answer extraction and checking tests
    test_skins.py       All 12 semantic skins coverage tests
```

---

## License

MIT. See [LICENSE](LICENSE).
