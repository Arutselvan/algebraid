"""
COMPEVAL Benchmark Runner - Evaluate gpt-4.1-nano on the COMPEVAL suite.
"""

import os
import json
import time
from compeval import CompEvalGenerator, CompEvalEvaluator
from compeval.llm_runner import run_tasks_on_llm

MODEL = "gpt-4.1-nano"
SEED = 42
DEPTHS = [1, 2, 3, 4]
TASKS_PER_DEPTH = 15
FAMILIES = ["intra", "inter", "field", "rule"]

print("=" * 60)
print(f"  COMPEVAL Benchmark — {MODEL}")
print("=" * 60)

# Step 1: Generate tasks
print(f"\n[1/3] Generating tasks (seed={SEED}, depths={DEPTHS}, {TASKS_PER_DEPTH} per depth)...")
gen = CompEvalGenerator(seed=SEED)
task_set = gen.generate(
    depths=DEPTHS,
    tasks_per_depth=TASKS_PER_DEPTH,
    families=FAMILIES,
    include_dimensions=True,
)
print(task_set.summary())
print(f"  Total tasks: {len(task_set)}")

# Save the task set
os.makedirs("results", exist_ok=True)
task_set.to_jsonl("results/task_set.jsonl")
print("  Task set saved to results/task_set.jsonl")

# Step 2: Run LLM
print(f"\n[2/3] Running {MODEL} on {len(task_set)} tasks...")
start_time = time.time()
predictions = run_tasks_on_llm(
    task_set,
    model=MODEL,
    temperature=0.0,
    max_tokens=512,
    delay=0.3,
    verbose=True,
)
elapsed = time.time() - start_time
print(f"  Completed in {elapsed:.1f}s")

# Save raw predictions
with open("results/predictions.json", "w") as f:
    json.dump(predictions, f, indent=2)
print("  Predictions saved to results/predictions.json")

# Step 3: Evaluate
print(f"\n[3/3] Evaluating predictions...")
evaluator = CompEvalEvaluator()
report = evaluator.evaluate(task_set, predictions, model_name=MODEL)
report.print_summary()

# Save report
report.save("results/eval_report.json")
print("  Report saved to results/eval_report.json")

# Save a human-readable summary
with open("results/summary.txt", "w") as f:
    import sys
    from io import StringIO
    old_stdout = sys.stdout
    sys.stdout = buf = StringIO()
    report.print_summary()
    sys.stdout = old_stdout
    f.write(buf.getvalue())
print("  Summary saved to results/summary.txt")

print("\nDone!")
