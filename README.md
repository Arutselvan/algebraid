# AlgebraicMirror 🪞

**A procedurally generated benchmark for compositional generalization using formal algebraic structures.**

> *"Can a language model reason through a chain of algebraic operations, or does it merely pattern-match the surface form?"*

---

## What is AlgebraicMirror?

AlgebraicMirror (COMPEVAL v2) is a benchmark that tests whether large language models can **compose operations** — not just recall facts. It generates tasks from four algebraic primitives (Z_n, S_n, D_n, GF(p)) and two composition operators (function chaining, direct product), producing tasks that are:

- **Procedurally generated** — infinite variety, no memorization possible
- **Symbolically verified** — every answer is computed from first principles
- **Multi-dimensional** — tests systematicity, substitutivity, productivity, and overgeneralization
- **Algebraically grounded** — four native complexity metrics unavailable in any other benchmark

## Project Structure

```
algebraicmirror/
├── compeval/                    # Core library
│   ├── primitives/              # Algebraic structures
│   │   ├── base.py              # Abstract base class
│   │   ├── cyclic_group.py      # Z_n
│   │   ├── symmetric_group.py   # S_n
│   │   ├── dihedral_group.py    # D_n
│   │   └── finite_field.py      # GF(p)
│   ├── composers/               # Composition operators
│   │   ├── direct_product.py    # G × H
│   │   └── function_composition.py  # f∘g∘h chains
│   ├── tasks/                   # Task utilities
│   │   ├── verbalizer.py        # 50+ prompt templates
│   │   └── verifier.py          # Answer extraction & checking
│   ├── generator.py             # Task generation (4 families)
│   ├── evaluator.py             # Multi-dimensional evaluation
│   ├── complexity.py            # Algebraic complexity metrics
│   ├── llm_runner.py            # OpenAI-compatible API runner
│   └── task_model.py            # Task/TaskSet data models
├── results/                     # Evaluation results
│   ├── eval_report.json         # Full structured report
│   ├── predictions.json         # Raw model responses
│   ├── task_set.jsonl           # All 296 generated tasks
│   ├── compeval_results.png     # Visualization charts
│   └── COMPEVAL_Report_gpt4_1_nano.md  # Human-readable report
├── run_benchmark.py             # Main benchmark runner
└── plot_results.py              # Results visualization
```

## Task Families

| Family | Description |
|--------|-------------|
| **Intra-Structure** | Chain d operations on one group element |
| **Inter-Structure** | Apply operations to tuples across d+1 nested groups |
| **Field Arithmetic** | Evaluate a depth-d expression tree in GF(p) |
| **Rule Induction** | Infer a hidden d-step chain from input-output examples |

## Algebraic Complexity Metrics

Four native metrics that only this benchmark can compute:

| Metric | Description |
|--------|-------------|
| **H_alg** | Algebraic Entropy — log₂(\|G\|) × depth |
| **D_comm** | Commutativity Distance — fraction of order-dependent op pairs |
| **O_c** | Orbit Complexity — fraction of group elements visited |
| **I_s** | Structural Interference — shared divisors in direct products |

## Results: `gpt-4.1-nano`

| Depth | Accuracy |
|-------|----------|
| 1 | 61.7% |
| 2 | 35.2% |
| 3 | 19.3% |
| 4 | 26.7% |
| **Overall** | **34.1%** |

See [`results/COMPEVAL_Report_gpt4_1_nano.md`](results/COMPEVAL_Report_gpt4_1_nano.md) for the full analysis.

## Quick Start

```python
from compeval import CompEvalGenerator, CompEvalEvaluator
from compeval.llm_runner import run_tasks_on_llm

gen = CompEvalGenerator(seed=42)
task_set = gen.generate(depths=[1, 2, 3, 4], tasks_per_depth=15)

predictions = run_tasks_on_llm(task_set, model="gpt-4.1-nano")

evaluator = CompEvalEvaluator()
report = evaluator.evaluate(task_set, predictions, model_name="gpt-4.1-nano")
report.print_summary()
```

## Requirements

```
openai
```

Set your `OPENAI_API_KEY` environment variable before running.
