# ALGEBRAID

**A procedurally generated benchmark for testing the compositional reasoning of AI models using formal algebraic structures.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

`algebraid` is a tool for researchers and developers to measure a model's ability to reason through multi-step algebraic problems. It moves beyond static benchmarks by procedurally generating tasks from a set of core algebraic primitives and composition rules. This ensures that models cannot simply memorize answers but must demonstrate genuine compositional generalization.

## What's New in v2.1

Version 2.1 addresses critical quality issues identified during external review and introduces a built-in validation layer:

- **Fixed power operation semantics**: For additive groups (Z_n, GF(p)), `power_k(x)` computes scalar multiplication `k*x mod n`, not iterated addition `x + k`. All skin descriptions now accurately say "multiply the current value by k" instead of the misleading "advance k times".
- **Fixed missing information in prompts**: Symmetric and dihedral group skins now provide explicit descriptions for power operations (e.g., "compose the current arrangement with itself 2 times") instead of falling through to generic placeholders like "apply a shuffle".
- **Fixed element labeling**: S_n elements are 1-indexed. Card and seating skins now correctly map `1 -> Card 1` / `1 -> Alice` instead of using off-by-one mappings that produced confusing labels like "Two, Three, Four" for S_3.
- **Built-in task validation**: New `algebraid validate` CLI command and `TaskValidator` API to catch prompt-answer misalignment, generic fallbacks, and structural issues before evaluation.

## Key Features

- **Procedural Generation**: Creates a virtually infinite number of unique tasks from four algebraic primitives (`CyclicGroup`, `SymmetricGroup`, `DihedralGroup`, `FiniteField`) and two composition operators (`FunctionComposition`, `DirectProduct`).
- **Semantic Skins**: 10 narrative skins translate abstract algebra into coherent real-world scenarios (Clock Arithmetic, Musical Intervals, Robot Steps, Color Wheel, Deck of Cards, Seating Arrangements, Polygon Symmetries, Tile Flips, Secret Codes, Modular Arithmetic).
- **Multi-Dimensional Evaluation**: Measures accuracy across four task families (intra-structure, inter-structure, field arithmetic, rule induction) and four Hupkes-style compositionality dimensions (systematicity, substitutivity, productivity, overgeneralization).
- **Algebraic Complexity Metrics**: Computes four native complexity metrics: Algebraic Entropy (H_alg), Commutativity Distance (D_comm), Orbit Complexity (O_c), and Structural Interference (I_s).
- **Built-in Validation**: Post-generation validation catches prompt-answer misalignment, missing parameters, and formatting issues.
- **Extensible Adapter Architecture**: Evaluate any model via adapters for OpenAI, Anthropic, HuggingFace, and custom HTTP endpoints.
- **Command-Line Interface**: CLI commands to `generate`, `run`, `evaluate`, and `validate` task sets.

## Quick Start

### 1. Installation

```bash
pip install algebraid
```

### 2. Set API Key

Export your OpenAI API key (other providers supported via adapters).

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 3. Run the Benchmark

```bash
# Step 1: Generate a task set (auto-validates by default)
algebraid generate --output ./data/my_tasks.jsonl --seed 42

# Step 2: Run the tasks against a model
algebraid run ./data/my_tasks.jsonl --output ./results/my_preds.json --model gpt-4.1-nano

# Step 3: Evaluate the predictions
algebraid evaluate ./data/my_tasks.jsonl ./results/my_preds.json --model-name "GPT-4.1-Nano"
```

## CLI Usage

### `algebraid generate`

Generate a new task set from scratch. Automatically validates the generated tasks unless `--skip-validation` is set.

| Argument | Default | Description |
|---|---|---|
| `-o`, `--output` | `./data/task_set.jsonl` | Output path for the task set. |
| `--seed` | `42` | Random seed for generation. |
| `--depths` | `1 2 3 4` | List of composition depths to generate. |
| `--tasks-per-depth` | `50` | Number of tasks per depth/family. |
| `--families` | `intra inter field rule` | Task families to include. |
| `--no-dims` | `False` | Exclude Hupkes dimensions. |
| `--skip-validation` | `False` | Skip post-generation validation. |

### `algebraid run`

Run a task set against a model using a specified adapter.

| Argument | Default | Description |
|---|---|---|
| `task_set` | | Path to the task set JSONL file. |
| `-o`, `--output` | `./results/predictions.json` | Output path for predictions. |
| `-a`, `--adapter` | `openai` | Adapter: `openai`, `anthropic`, `huggingface`, `custom_http`. |
| `-m`, `--model` | `gpt-4.1-nano` | Name of the model to run. |
| `-t`, `--temperature` | `0.0` | Sampling temperature. |
| `--max-tokens` | `512` | Max tokens for the model response. |
| `--delay` | `0.5` | Delay between API calls (seconds). |

### `algebraid evaluate`

Evaluate a set of model predictions against a task set.

| Argument | Default | Description |
|---|---|---|
| `task_set` | | Path to the task set JSONL file. |
| `predictions` | | Path to the predictions JSON file. |
| `-o`, `--output` | `./results/report.json` | Output path for the evaluation report. |
| `--model-name` | `unknown_model` | Name of the model for the report. |
| `--strict` | `False` | Enable strict answer matching. |

### `algebraid validate`

Validate a task set for quality issues (prompt completeness, answer consistency, etc.).

| Argument | Default | Description |
|---|---|---|
| `task_set` | | Path to the task set JSONL file. |
| `-o`, `--output` | `None` | Output path for validation report JSON. |

## Python API

```python
from algebraid import AlgebraidGenerator, AlgebraidEvaluator, TaskValidator

# 1. Generate tasks
gen = AlgebraidGenerator(seed=42)
task_set = gen.generate(depths=[1, 2, 3], tasks_per_depth=20)

# 2. Validate tasks
validator = TaskValidator()
report = validator.validate_taskset(task_set)
print(f"Validation: {report['passed']}/{report['total_tasks']} passed")

# 3. Run tasks (requires API key)
from algebraid.adapters import get_adapter
Adapter = get_adapter("openai")
adapter = Adapter(model="gpt-4.1-nano", temperature=0.0, max_tokens=512, delay=0.5, verbose=True)
predictions = adapter.run_tasks(task_set)

# 4. Evaluate
evaluator = AlgebraidEvaluator()
eval_report = evaluator.evaluate(task_set, predictions, model_name="GPT-4.1-Nano")
eval_report.print_summary()
eval_report.save("report.json")
```

## Semantic Skins

Semantic skins translate abstract algebraic operations into coherent real-world narratives, testing whether models can reason through the same mathematical structure when presented in different surface forms.

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

### Operation Semantics by Structure Type

Understanding how the `power_k` operation maps to each structure type is critical for prompt accuracy:

| Structure | `op(a, b)` | `power_k(x)` | Skin Description |
|---|---|---|---|
| Z_n (additive) | `a + b mod n` | `k * x mod n` (scalar multiplication) | "multiply the current value by k" |
| GF(p) (additive) | `a + b mod p` | `k * x mod p` (scalar multiplication) | "multiply by k" |
| S_n (composition) | `a composed with b` | `x composed with itself k times` | "compose the current arrangement with itself k times" |
| D_n (composition) | `a * b` in D_n | `x * x * ... * x` (k times) | "compose the current transformation with itself k times" |

## Architecture

```
algebraid/
  primitives/          # Algebraic structures: Z_n, S_n, D_n, GF(p)
  composers/           # Composition: FunctionComposition, DirectProduct
  tasks/               # Verbalizer, Verifier, Validator
  skins.py             # 10 semantic narrative skins
  generator.py         # Procedural task generation
  evaluator.py         # Evaluation and reporting
  complexity.py        # Algebraic complexity metrics
  adapters.py          # Model adapter architecture
  cli.py               # Command-line interface
```

## Copyright

Copyright (c) 2025 Arut Selvan Dhanasekaran. All rights reserved.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
