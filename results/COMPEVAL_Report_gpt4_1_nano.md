# COMPEVAL Evaluation Report — `gpt-4.1-nano`

**Date:** February 24, 2026  
**Benchmark Version:** COMPEVAL v2.0 (seed=42)  
**Model Evaluated:** `gpt-4.1-nano`  
**Total Tasks:** 296  

---

## 1. Overall Results

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **34.1%** (101 / 296) |
| Compositional Ceiling @ 50% | Depth 1 |
| Compositional Ceiling @ 25% | Depth 4 |

The model achieves 61.7% accuracy at depth 1 (single-operation tasks), but degrades sharply as composition depth increases, falling to 19.3% at depth 3 before a slight recovery at depth 4. This confirms a strong **depth-degradation effect** — the model struggles to maintain compositional reasoning across longer operation chains.

---

## 2. Accuracy by Composition Depth

| Depth | Correct | Total | Accuracy |
|-------|---------|-------|----------|
| 1 | 37 | 60 | **61.7%** |
| 2 | 31 | 88 | **35.2%** |
| 3 | 17 | 88 | **19.3%** |
| 4 | 16 | 60 | **26.7%** |

The depth-degradation curve shows a clear decline from depth 1 to depth 3. The slight uptick at depth 4 may be due to the model defaulting to simpler heuristics that happen to coincide with correct answers for some task families.

---

## 3. Accuracy by Task Family

| Family | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| **Field Arithmetic** | 46 | 60 | **76.7%** |
| Inter-Structure Composition | 20 | 74 | **27.0%** |
| Intra-Structure Composition | 23 | 102 | **22.5%** |
| Rule Induction | 12 | 60 | **20.0%** |

**Field arithmetic** is the strongest family — the model performs well on modular arithmetic in GF(p), likely because it resembles standard integer arithmetic that appears frequently in training data. **Rule induction** is the weakest, requiring the model to infer a hidden function from examples, which demands a higher level of abstract reasoning.

---

## 4. Accuracy by Hupkes Compositionality Dimension

| Dimension | Correct | Total | Accuracy |
|-----------|---------|-------|----------|
| General | 88 | 240 | **36.7%** |
| Systematicity | 7 | 28 | **25.0%** |
| Substitutivity | 3 | 14 | **21.4%** |
| Overgeneralization | 3 | 14 | **21.4%** |

The model performs worst on **substitutivity** and **overgeneralization** tasks. Substitutivity tasks use relabelled elements (e.g., city names instead of integers), which breaks the model's reliance on surface-form pattern matching. Overgeneralization tasks involve non-commutative groups where the model incorrectly assumes commutativity (a∘b = b∘a), a known failure mode.

---

## 5. Algebraic Complexity Metrics

These four metrics are unique to COMPEVAL and are computed from the algebraic structure of each task — they cannot be obtained from standard NLP benchmarks.

| Metric | Description | Average Value |
|--------|-------------|---------------|
| **H_alg** (Algebraic Entropy) | Information-theoretic difficulty; log₂(\|G\|) × depth | 7.7579 |
| **D_comm** (Commutativity Distance) | Fraction of consecutive op pairs where order matters | 0.1723 |
| **O_c** (Orbit Complexity) | Fraction of group elements visited by the computation trace | 0.0485 |
| **I_s** (Structural Interference) | Shared divisors between component group orders in direct products | 0.0822 |

The high **H_alg** value (7.76 bits on average) reflects the large search spaces involved. The low **D_comm** (0.17) indicates that most tasks in this benchmark use commutative structures, which is consistent with the prevalence of CyclicGroup and FiniteField tasks. The low **O_c** (0.05) shows that most computations visit only a small fraction of the group, meaning the model needs to track only a few distinct states — yet still struggles.

---

## 6. Key Findings

1. **Sharp depth degradation**: Accuracy drops from 61.7% at depth 1 to 19.3% at depth 3, confirming that `gpt-4.1-nano` does not compose operations reliably beyond 2–3 steps.

2. **Field arithmetic is a strength**: The model handles modular arithmetic in GF(p) well (76.7%), likely due to familiarity with integer arithmetic.

3. **Non-commutativity is a weakness**: Overgeneralization accuracy (21.4%) shows the model frequently assumes commutativity even in non-commutative groups like S₃ and D₄.

4. **Substitutivity is brittle**: Relabelling elements with non-numeric names (cities, colors) reduces accuracy from 36.7% to 21.4%, revealing surface-form dependence.

5. **Rule induction is hardest**: At 20.0%, the model struggles to infer hidden algebraic functions from examples, even at low depths.

---

## 7. Files

| File | Description |
|------|-------------|
| `results/eval_report.json` | Full structured evaluation report (JSON) |
| `results/predictions.json` | Raw model responses for all 296 tasks |
| `results/task_set.jsonl` | All generated tasks with ground-truth answers |
| `results/compeval_results.png` | Visualization charts |
| `results/summary.txt` | Plain-text console summary |
