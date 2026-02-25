# ALGEBRAID 🧠

**A procedurally generated benchmark for testing the compositional reasoning of AI models using formal algebraic structures.**

[![PyPI version](https://badge.fury.io/py/algebraid.svg)](https://badge.fury.io/py/algebraid)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

`algebraid` is a tool for researchers and developers to measure a model's ability to reason through multi-step algebraic problems. It moves beyond static benchmarks by procedurally generating tasks from a set of core algebraic primitives and composition rules. This ensures that models cannot simply memorize answers but must demonstrate genuine compositional generalization.

## Key Features

- **Procedural Generation**: Creates a virtually infinite number of unique tasks from four algebraic primitives (`CyclicGroup`, `SymmetricGroup`, `DihedralGroup`, `FiniteField`) and two composition operators (`FunctionComposition`, `DirectProduct`).
- **Multi-Dimensional Evaluation**: Measures accuracy across four task families (intra-structure, inter-structure, field arithmetic, rule induction) and four Hupkes-style compositionality dimensions (systematicity, substitutivity, productivity, overgeneralization).
- **Algebraic Complexity Metrics**: Computes four native complexity metrics unique to this benchmark: Algebraic Entropy (H_alg), Commutativity Distance (D_comm), Orbit Complexity (O_c), and Structural Interference (I_s).
- **Extensible Adapter Architecture**: Easily evaluate any model via a simple adapter system. Support for OpenAI, Anthropic, HuggingFace, and custom HTTP endpoints is planned.
- **Command-Line Interface**: A simple and powerful CLI to `generate` task sets, `run` them against a model, and `evaluate` the results.

## Quick Start

### 1. Installation

```bash
pip install algebraid
```

### 2. Set API Key

Export your OpenAI API key (other providers will be supported soon).

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 3. Run the Benchmark

Run the full benchmark pipeline with a single command.

```bash
# Step 1: Generate a task set
algebraid generate --output ./data/my_tasks.jsonl --seed 42

# Step 2: Run the tasks against a model
algebraid run ./data/my_tasks.jsonl --output ./results/my_preds.json --model gpt-4.1-nano

# Step 3: Evaluate the predictions
algebraid evaluate ./data/my_tasks.jsonl ./results/my_preds.json --model-name "GPT-4.1-Nano"
```

This will produce a detailed evaluation report in `./results/report.json` and print a summary to the console.

## CLI Usage

### `algebraid generate`

Generate a new task set from scratch.

| Argument | Default | Description |
|---|---|---|
| `-o`, `--output` | `./data/task_set.jsonl` | Output path for the task set. |
| `--seed` | `42` | Random seed for generation. |
| `--depths` | `1 2 3 4` | List of composition depths to generate. |
| `--tasks-per-depth` | `50` | Number of tasks per depth/family. |
| `--families` | `intra inter field rule` | Task families to include. |
| `--no-dims` | `False` | Exclude Hupkes dimensions. |

### `algebraid run`

Run a task set against a model using a specified adapter.

| Argument | Default | Description |
|---|---|---|
| `task_set` | | Path to the task set JSONL file. |
| `-o`, `--output` | `./results/predictions.json` | Output path for predictions. |
| `-a`, `--adapter` | `openai` | Adapter to use (e.g., `openai`, `anthropic`). |
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

## Python API

For more advanced use cases, you can use the Python API directly.

```python
from algebraid import CompEvalGenerator, CompEvalEvaluator
from algebraid.adapters import get_adapter

# 1. Generate tasks
gen = CompEvalGenerator(seed=42)
task_set = gen.generate(depths=[1, 2], tasks_per_depth=10)

# 2. Run tasks
Adapter = get_adapter("openai")
adapter = Adapter(model="gpt-4.1-nano", temperature=0.0, max_tokens=512, delay=0.5, verbose=True)
predictions = adapter.run_tasks(task_set)

# 3. Evaluate
evaluator = CompEvalEvaluator()
report = evaluator.evaluate(task_set, predictions, model_name="GPT-4.1-Nano")
report.print_summary()

# Save the report
report.save("report.json")
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
