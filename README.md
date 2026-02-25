# ALGEBRAID

**A procedurally generated benchmark for testing compositional reasoning in language models using formal algebraic structures.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

ALGEBRAID generates an unlimited supply of compositional reasoning tasks grounded in four algebraic primitives and ten narrative skins. Because every task is produced from a formal specification, ground-truth answers are guaranteed to be correct and the benchmark cannot be memorised.

## Features

ALGEBRAID provides procedural generation from four algebraic primitives (`CyclicGroup`, `SymmetricGroup`, `DihedralGroup`, `FiniteField`) combined through two composition operators (`FunctionComposition`, `DirectProduct`). Ten semantic skins translate abstract algebra into coherent real-world scenarios, ensuring that models are tested on reasoning rather than pattern matching.

Evaluation spans four task families (intra-structure, inter-structure, field arithmetic, rule induction) and four Hupkes-style compositionality dimensions (systematicity, substitutivity, productivity, overgeneralization). Four algebraic complexity metrics provide fine-grained analysis: Algebraic Entropy, Commutativity Distance, Orbit Complexity, and Structural Interference.

A built-in validation layer checks every generated task for structural soundness, prompt completeness, and answer consistency before it is written to disk.

## Quick Start

```bash
pip install -e .
```

```bash
# Generate a task set (auto-validated)
algebraid generate --seed 42

# Validate an existing task set
algebraid validate ./data/algebraid_s42_20260225.jsonl

# Run tasks against a model (requires OPENAI_API_KEY)
algebraid run ./data/algebraid_s42_20260225.jsonl -m gpt-4.1-nano

# Evaluate predictions
algebraid evaluate ./data/algebraid_s42_20260225.jsonl ./results/predictions.json
```

## CLI Reference

### `algebraid generate`

| Argument | Default | Description |
|---|---|---|
| `-o`, `--output` | `./data/algebraid_s{seed}_{date}.jsonl` | Output JSONL path. |
| `--seed` | `42` | Random seed. |
| `--depths` | `1 2 3 4` | Composition depths. |
| `--tasks-per-depth` | `50` | Tasks per depth/family combination. |
| `--families` | `intra inter field rule` | Task families to include. |
| `--no-dims` | | Exclude Hupkes compositionality dimensions. |
| `--skip-validation` | | Skip post-generation validation. |

### `algebraid run`

| Argument | Default | Description |
|---|---|---|
| `task_set` | | Path to the task set JSONL. |
| `-o`, `--output` | `./results/predictions.json` | Output predictions JSON. |
| `-a`, `--adapter` | `openai` | Adapter: `openai`, `anthropic`, `huggingface`, `custom_http`. |
| `-m`, `--model` | `gpt-4.1-nano` | Model identifier. |
| `-t`, `--temperature` | `0.0` | Sampling temperature. |
| `--max-tokens` | `512` | Maximum response tokens. |
| `--delay` | `0.5` | Delay between API calls (seconds). |

### `algebraid evaluate`

| Argument | Default | Description |
|---|---|---|
| `task_set` | | Path to the task set JSONL. |
| `predictions` | | Path to the predictions JSON. |
| `-o`, `--output` | `./results/report.json` | Output report JSON. |
| `--model-name` | `unknown` | Model display name for the report. |
| `--strict` | | Enable strict answer matching. |

### `algebraid validate`

| Argument | Default | Description |
|---|---|---|
| `task_set` | | Path to the task set JSONL. |
| `-o`, `--output` | | Output validation report JSON. |

## Python API

```python
from algebraid import AlgebraidGenerator, AlgebraidEvaluator, TaskValidator

# Generate
gen = AlgebraidGenerator(seed=42)
task_set = gen.generate(depths=[1, 2, 3], tasks_per_depth=20)

# Validate
report = TaskValidator().validate_set(task_set)
print(f"{report['passed']}/{report['total']} tasks passed validation")

# Save
task_set.to_jsonl("./data/algebraid_s42.jsonl")
```

## Semantic Skins

| Skin | Structure | Domain |
|---|---|---|
| Clock Arithmetic | Z_n | Positions on an n-hour clock |
| Musical Intervals | Z_n | Tones on a chromatic scale |
| Robot Steps | Z_n | Stops on a circular track |
| Color Wheel | Z_n | Hues on a color wheel |
| Deck of Cards | S_n | Shuffles of n cards |
| Seating Arrangements | S_n | Rearrangements of n people |
| Polygon Symmetries | D_n | Symmetries of a regular polygon |
| Tile Flips and Rotations | D_n | Orientations of a regular tile |
| Secret Codes | GF(p) | Symbols in a code system |
| Modular Arithmetic | GF(p) | Residues modulo a prime |

### Operation Semantics

| Structure | `op(a, b)` | `power_k(x)` | Skin phrasing |
|---|---|---|---|
| Z_n | `a + b mod n` | `k * x mod n` | "multiply the current value by k" |
| GF(p) | `a + b mod p` | `k * x mod p` | "multiply by k" |
| S_n | `a composed with b` | `x composed k times` | "compose with itself k times" |
| D_n | `a * b` in D_n | `x^k` in D_n | "compose with itself k times" |

## Project Layout

```
src/algebraid/
    primitives/      Algebraic structures (Z_n, S_n, D_n, GF(p))
    composers/       Composition operators (FunctionComposition, DirectProduct)
    tasks/           Verbalization, verification, and validation
    skins.py         Ten semantic narrative skins
    generator.py     Procedural task generation
    evaluator.py     Evaluation and reporting
    complexity.py    Algebraic complexity metrics
    adapters.py      Model adapter architecture
    cli.py           Command-line interface
```

## License

MIT. See [LICENSE](LICENSE).
