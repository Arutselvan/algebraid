# ALGEBRAID

**A procedurally generated benchmark for compositional algebraic reasoning in language models.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

ALGEBRAID generates an unlimited, non-memorisable supply of reasoning tasks built on five algebraic structures. Every answer is derived by algebraic construction; chain-computation answers (intra-structure, adversarial, intermediate) are additionally re-derived by an independent proof engine from the solution trace. Other families (inter-structure, conceptual, rule induction) are verified by construction rather than by trace. Tasks are seed-deterministic: the same seed always produces the same task set.

The proof engine verifies **algebraic trace consistency** — it re-applies each named operation from first principles and checks every intermediate value. It does not re-read the prompt. A separate **prompt-trace alignment check** in the validator warns when a trace operation references an element that does not appear literally in the prompt (catches verbalizer bugs that the proof engine cannot detect).

## Key Properties

| Property | Detail |
|---|---|
| **Correct by construction** | All answers derived by computation, not annotation |
| **Independently verified** | Proof engine re-derives answers for all chain tasks (intra, adversarial, intermediate) from the solution trace; other families are verified by construction |
| **Non-memorisable** | Unlimited procedural variation; no fixed test set |
| **Reproducible** | Same seed -> same task set |
| **Multi-dimensional** | 7 generator families (5 task family labels) x 7 compositional dimensions x 4 complexity metrics |
| **Skin metadata** | Active skin name stored in task `metadata["skin"]` for downstream analysis |

## Install

```bash
pip install -e .
```

No provider SDK is bundled. Install only what you need:

```bash
pip install -e ".[openai]"      # OpenAI API and custom HTTP endpoints
pip install -e ".[anthropic]"   # Anthropic API
```

Set the corresponding API key before running:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
```

## Quick Start

```bash
# Generate a task set (validated automatically)
algebraid generate --seed 42
# -> ./data/algebraid_s42_20260225_e50604.jsonl

# Full pipeline: run model, evaluate, prove, analyze
algebraid pipeline ./data/algebraid_s42_20260225_e50604.jsonl \
    --model gpt-4.1-nano --adapter openai
# -> ./runs/20260225_143012_gpt_4_1_nano_algebraid_s42_.../
```

Full CLI reference: [docs/cli.md](docs/cli.md)

---

## Task Families

Seven generator families are available via `--families` (all seven are included by default):

| Family key | Family label | Description |
|---|---|---|
| `intra` | intra-structure composition | Composition chain within a single structure |
| `inter` | inter-structure composition | Component-wise operation across a direct product |
| `field` | field arithmetic | Expression evaluation in a finite field |
| `rule` | rule induction | Identify the pattern; give the next element |
| `conceptual` | conceptual query | Structural property query: identity, order, commutativity |
| `adversarial` | intra-structure composition | Intra-structure chain designed to trigger specific reasoning errors |
| `intermediate` | intra-structure composition | Report the value after step k of an n-step chain |

`adversarial` and `intermediate` tasks are generated as intra-structure chains. They share the `intra-structure composition` family label in the task data and appear under that label in `accuracy_by_family` reports. They are distinguished from standard intra tasks by their `dimension` tag (`adversarial` or `intermediate_state`), which appears in `accuracy_by_dimension` reports.

### Adversarial Sub-types

| Sub-type | Trap |
|---|---|
| `double_inverse` | Two inverses cancel; answer equals the starting value |
| `self_cancelling` | Operations sum to zero mod n |
| `identity_bait` | Result is the identity element; common wrong answer is the starting value |
| `commutativity_trap` | Non-abelian structure; answer depends on operand order |

---

## Algebraic Structures

| Structure | Notation | Abelian | Order |
|---|---|---|---|
| Cyclic group | Z_n, n in [3, 18] | Yes | n |
| Symmetric group | S_n, n in {3, 4, 5} | No | n! |
| Dihedral group | D_n, n in [3, 8] | No | 2n |
| Finite field | GF(p), p in {3,5,7,11,13,17,19,23} | Yes | p |
| Quaternion group | Q_8 | No | 8 |

---

## Semantic Skins

Each task can be expressed through a narrative skin that maps abstract algebra to a real-world context. Skins are applied at generation time and affect prompt wording only, not the underlying computation.

| Skin | Structure | Domain |
|---|---|---|
| Clock Arithmetic | Z_n | Positions on an n-hour clock |
| Musical Intervals | Z_n | Tones on a chromatic scale |
| Robot Steps | Z_n | Stops on a circular track |
| Color Wheel | Z_n | Hues on a color wheel |
| Deck of Cards | S_n | Shuffles of n cards |
| Seating Arrangements | S_n | Rearrangements of n people |
| Polygon Symmetries | D_n | Symmetries of a regular polygon |
| Tile Flips and Rotations | D_n | Orientations of a tile |
| Secret Codes | GF(p) | Symbols in a code system |
| Modular Arithmetic | GF(p) | Residues modulo a prime |
| Quaternion Algebra | Q_8 | Quaternion unit arithmetic |
| Quaternion Rotations | Q_8 | 3-D rotation states |

---

## Compositional Dimensions

Every task is tagged with one compositional dimension. The first four follow Hupkes et al. (2020); the remaining three are ALGEBRAID-specific. Evaluation reports break down accuracy by dimension via `accuracy_by_dimension`.

| Dimension | Source | Tests |
|---|---|---|
| `systematicity` | Hupkes et al. (2020) | Unseen combination of seen components |
| `substitutivity` | Hupkes et al. (2020) | Synonym substitution invariance |
| `productivity` | Hupkes et al. (2020) | Longer chain than seen during training |
| `overgeneralization` | Hupkes et al. (2020) | Resistance to applying rules where they don't hold |
| `adversarial` | ALGEBRAID | Resistance to specific reasoning shortcuts |
| `intermediate_state` | ALGEBRAID | Access to intermediate computation steps |
| `general` | ALGEBRAID | Standard task; no specific dimension stress |

---

## Algebraic Complexity Metrics

Every task carries four algebraic complexity scores computed independently of the model:

| Metric | Symbol | Measures |
|---|---|---|
| Algebraic Entropy | H_alg | log_2(\|G\|) x depth (intra); log_2(∑\|G_i\|) (inter); 0 (conceptual) |
| Commutativity Distance | D_comm | Fraction of consecutive operation pairs where operand order matters |
| Orbit Complexity | O_c | Distinct intermediate values / \|G\|: breadth of element traversal |
| Structural Interference | I_s | Coprimality-based interference between direct product components |

Two task-specific metrics are available as standalone functions:

| Function | Applies to | Range |
|---|---|---|
| `compute_conceptual_depth(task)` | Conceptual queries | 0.1 (identity lookup) to 0.7 (element order) |
| `compute_adversarial_strength(task)` | Adversarial tasks | 0.3 (self-cancelling) to 0.8 (commutativity trap) |

---

## Scaling Law Analysis

`run_analysis(report)` returns a pooled power-law fit `acc(d) ~ A * d^(-alpha)` across all chain families. **Depth semantics differ by family** — intra-structure depth is chain length, inter-structure depth is component count, field arithmetic depth is expression-tree height — so the pooled fit mixes incommensurable units.

Use `fit_scaling_law_by_family(report)` for per-family independent fits, which avoids conflating these semantics. Each family result includes:

| Key | Meaning |
|---|---|
| `alpha` | Decay exponent; larger = steeper drop with depth |
| `alpha_se` | Standard error of the exponent estimate |
| `r2` | R² of the log-log fit |
| `interpretation` | "Strong" (R²≥0.95), "Moderate" (R²≥0.80), "Weak" (<0.80), or "Accuracy does not decrease..." (alpha≤0) |
| `note` | Data-quality warning if fewer than 8 depth levels are available |

## Proof Report Keys

`verify_set(task_set)` returns a dict with the following keys:

| Key | Meaning |
|---|---|
| `total` | Total tasks |
| `trace_verified` | Tasks with traces that re-derive correctly from first principles |
| `skipped` | Tasks without traces (conceptual, rule induction) — verified by construction |
| `failed` | Tasks where the trace is algebraically inconsistent |
| `proof_rate` | `trace_verified / (trace_verified + failed)` as a percentage |
| `coverage` | `(trace_verified + skipped) / total` as a percentage |
| `failures` | List of `{task_id, failed_step, message}` dicts |

## Python API

```python
from algebraid import (
    AlgebraidGenerator, AlgebraidEvaluator, EvalReport,
    TaskValidator, verify_set, run_analysis, fit_scaling_law_by_family,
    split_by_depth, split_by_commutativity,
)

# Generate
# API defaults: depths=[1,2,3,4,5], tasks_per_depth=10, families=all seven
gen = AlgebraidGenerator(seed=42)
task_set = gen.generate(
    depths=[1, 2, 3, 4],
    tasks_per_depth=20,
    families=["intra", "inter", "conceptual", "adversarial"],
)

# Validate (includes prompt-trace alignment check)
val_report = TaskValidator().validate_set(task_set)
print(f"{val_report['passed']}/{val_report['total']} passed")

# Independently verify trace-based answers
proof = verify_set(task_set)
print(f"Trace-verified: {proof['trace_verified']}  "
      f"proof rate: {proof['proof_rate']}%")

# Evaluate (given a predictions dict)
predictions = {"AG-abc123": "5"}
evaluator = AlgebraidEvaluator()
report = evaluator.evaluate(task_set, predictions, model_name="my-model")
report.print_summary()

# Error analysis (pooled scaling law + per-family breakdown)
analysis = run_analysis(report)

# Per-family scaling law (recommended: avoids mixing incompatible depth semantics)
# intra: depth = chain length; inter: depth = #components; field: depth = tree height
by_family = fit_scaling_law_by_family(report)
for fam, fit in by_family.items():
    if fit["alpha"] is not None:
        print(f"{fam}: alpha={fit['alpha']:.2f} ±{fit['alpha_se']:.2f}  "
              f"R²={fit['r2']:.3f}  ({fit['interpretation']})")

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

## Project Layout

```
src/algebraid/
    primitives/         Z_n, S_n, D_n, GF(p), Q_8, abstract base
    composers/          FunctionComposition, DirectProduct
    tasks/              Verbalization, answer verification, schema validation
    task_model.py       Task, TaskSet, TaskFamily, CompositionDimension
    generator.py        Procedural task generation (7 families)
    evaluator.py        AlgebraidEvaluator, EvalReport
    proof.py            Independent algebraic verifier
    analysis.py         Scaling laws, phase transitions, error taxonomy; per-family fits via fit_scaling_law_by_family()
    complexity.py       Algebraic complexity metrics
    splits.py           Train/test split strategies
    skins.py            12 semantic skins
    adapters.py         OpenAI, Anthropic, HuggingFace, custom HTTP adapters
    cli.py              8-command CLI
tests/
    test_primitives.py  Group axiom verification for all 5 structures
    test_composers.py   Composition and direct product tests
    test_generator.py   Determinism, schema, and grading tests
    test_complexity.py  Algebraic complexity metric tests
    test_task_model.py  Task/TaskSet serialization round-trip tests
    test_verbalizer.py  Prompt generation and skin integration tests
    test_verifier.py    Answer extraction and checking tests
    test_skins.py       Semantic skin coverage tests
    test_evaluator.py   EvalReport scoring and ceiling tests
    test_analysis.py    Scaling law, phase transition, error taxonomy tests
    test_splits.py      Train/test split strategy tests
```

---

## License

MIT. See [LICENSE](LICENSE).
