# CLI Reference

All commands follow the pattern `algebraid <command> [options]`. Run `algebraid <command> --help` for the full argument list.

---

## generate

Create a task set and validate it immediately.

```bash
algebraid generate --seed 42 --depths 1 2 3 4 --tasks-per-depth 50
```

| Argument | Default | Description |
|---|---|---|
| `-o, --output` | `./data/algebraid_s{seed}_{date}_{hash}.jsonl` | Output path |
| `--seed` | `42` | Random seed |
| `--depths` | `1 2 3 4` | Composition depths to generate |
| `--tasks-per-depth` | `50` | Tasks per depth per family |
| `--families` | all seven | Space-separated list of generator families to include |
| `--skip-validation` | | Skip post-generation schema validation |

The 6-character config hash in the default filename is derived from `(depths, tasks_per_depth, families)`. Different generation parameters never collide even with the same seed and date.

---

## run

Run a task set against a model and save raw predictions.

```bash
algebraid run ./data/tasks.jsonl -m gpt-4.1-nano -a openai -q
```

| Argument | Default | Description |
|---|---|---|
| `task_set` | required | Path to task set JSONL |
| `-o, --output` | `./results/preds_{model}_{taskset}_{ts}.json` | Output predictions path |
| `-m, --model` | `gpt-4.1-nano` | Model identifier |
| `-a, --adapter` | `openai` | `openai` \| `anthropic` \| `huggingface` \| `custom_http` |
| `-t, --temperature` | `0.0` | Sampling temperature |
| `--max-tokens` | `512` | Max response tokens |
| `--delay` | `0.5` | Seconds between API calls |
| `-q, --quiet` | | Suppress per-task progress output |

---

## evaluate

Score an existing predictions file against a task set.

```bash
algebraid evaluate ./data/tasks.jsonl ./results/preds_*.json --model-name gpt-4.1-nano
```

| Argument | Default | Description |
|---|---|---|
| `task_set` | required | Path to task set JSONL |
| `predictions` | required | Path to predictions JSON |
| `-o, --output` | `./results/report_{model}_{taskset}_{ts}.json` | Output report path |
| `--model-name` | `"unknown"` | Display name for the report |
| `--strict` | | Strict answer matching |

---

## pipeline

End-to-end evaluation in one command: run the model, evaluate predictions, verify algebraic correctness, and compute error analysis. All artifacts are saved to a uniquely named folder.

```bash
algebraid pipeline ./data/tasks.jsonl --model claude-sonnet-4-6 --adapter anthropic
```

| Argument | Default | Description |
|---|---|---|
| `task_set` | required | Path to task set JSONL |
| `-m, --model` | `gpt-4.1-nano` | Model identifier |
| `-a, --adapter` | `openai` | `openai` \| `anthropic` \| `huggingface` \| `custom_http` |
| `-t, --temperature` | `0.0` | Sampling temperature |
| `--max-tokens` | `512` | Max response tokens |
| `--delay` | `0.5` | Seconds between API calls |
| `-q, --quiet` | | Suppress per-task progress output |
| `--strict` | | Strict answer matching |
| `--skip-prove` | | Skip algebraic proof verification |
| `--skip-analyze` | | Skip error scaling and phase analysis |
| `-o, --output-dir` | `./runs` | Base directory for run folders |

### Output Folder

The pipeline creates a uniquely named folder per run:

```
./runs/20260225_143012_gpt_4_1_nano_algebraid_s42_20260225_e50604/
    manifest.json       Run metadata, model config, accuracy summary, artifact paths
    predictions.json    Raw model responses keyed by task_id
    eval_report.json    Accuracy by depth / family / dimension; complexity metrics; per-task results
    proof_report.json   Algebraic verification: proven / skipped / failed counts  [skipped with --skip-prove]
    analysis.json       Power-law fit, phase transition, error taxonomy, hallucination onset  [skipped with --skip-analyze]
```

The folder name encodes the timestamp, model, and task set - runs are sortable and self-describing without opening any file.

`eval_report.json` stores up to 512 characters of each model response, so `algebraid analyze` can produce a complete error taxonomy from a saved report without re-running the model.

---

## validate

Check a task set for schema errors, prompt quality issues, and answer inconsistencies.

```bash
algebraid validate ./data/tasks.jsonl
```

| Argument | Default | Description |
|---|---|---|
| `task_set` | required | Path to task set JSONL |
| `-o, --output` | | Path to save validation report JSON (prints to stdout if omitted) |

---

## prove

Re-derive every answer from its solution trace using algebraic primitives, independently of the generator's computation path.

```bash
algebraid prove ./data/tasks.jsonl
```

| Argument | Default | Description |
|---|---|---|
| `task_set` | required | Path to task set JSONL |
| `-o, --output` | | Path to save proof report JSON (prints to stdout if omitted) |

---

## analyze

Compute error scaling laws, phase transitions, and mechanistic error taxonomy from an evaluation report.

```bash
algebraid analyze ./runs/20260225_143012_.../eval_report.json
```

| Argument | Default | Description |
|---|---|---|
| `report` | required | Path to an evaluation report JSON |
| `-o, --output` | | Path to save analysis JSON (prints to stdout if omitted) |

---

## split

Partition a task set into train/test splits for compositional generalisation studies.

```bash
# Split by composition depth
algebraid split ./data/tasks.jsonl --mode depth --train-max-depth 2

# Split by commutativity: abelian train, non-abelian test
algebraid split ./data/tasks.jsonl --mode commutativity

# Split by structure name prefix
algebraid split ./data/tasks.jsonl --mode structure \
    --train-prefixes Z_ GF( --test-prefixes S_ D_

# Split by task family or dimension
algebraid split ./data/tasks.jsonl --mode family \
    --train-families intra inter --test-families adversarial
```

| Argument | Default | Description |
|---|---|---|
| `task_set` | required | Path to task set JSONL |
| `--mode` | required | `depth` \| `commutativity` \| `structure` \| `family` |
| `--train-max-depth` | `2` | [depth] Max depth in train set |
| `--test-min-depth` | train+2 | [depth] Min depth in test set |
| `--train-prefixes` | | [structure] Structure name prefixes for train |
| `--test-prefixes` | | [structure] Structure name prefixes for test |
| `--train-families` | | [family] Family keys or dimension names for train |
| `--test-families` | | [family] Family keys or dimension names for test |
| `--output-dir` | `./data` | Directory for output split JSONL files |

The depth split creates a gap: tasks at depth `train_max_depth + 1` are excluded from both sets, so the test distribution strictly exceeds the training distribution.

---

## Output Naming Convention

All default output paths are self-describing:

| Artifact | Pattern |
|---|---|
| Task set | `./data/algebraid_s{seed}_{YYYYMMDD}_{config_hash}.jsonl` |
| Predictions | `./results/preds_{model}_{taskset}_{timestamp}.json` |
| Eval report | `./results/report_{model}_{taskset}_{timestamp}.json` |
| Run folder | `./runs/{YYYYMMDD_HHMMSS}_{model}_{taskset}/` |
