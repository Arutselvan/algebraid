# ALGEBRAID v2.1.0 Changelog

## Summary

Version 2.1.0 addresses all three critical defects identified in the external Gemini review of the task set, and introduces a built-in validation layer to prevent future regressions. All 312 generated tasks now pass validation with a 100% pass rate.

---

## Critical Defects Fixed

### 1. Misleading Power Operation Descriptions (Defect #1)

**Problem:** For additive groups (Z_n, GF(p)), the `power_k` operation computes `x + x + ... + x` (k times) = `k * x mod n` (scalar multiplication). However, the skin descriptions said things like "advance by one step, k times" or "ascend by one semitone, k times", which a human would interpret as `x + k` (iterated addition of 1), not `k * x`.

**Example:** Z_5, x=1, power_2. The old description "ascend by one semitone, 2 times" leads a human to compute 1 → 2 → 3 (answer: 3). But the actual answer is 2*1 mod 5 = 2.

**Fix:** All 10 skins now use mathematically accurate descriptions:
- CyclicGroup skins: "multiply the current position by k (mod n)"
- FiniteField skins: "multiply the current code value by k (mod p)"
- SymmetricGroup skins: "compose the current arrangement with itself k times"
- DihedralGroup skins: "compose the current transformation with itself k times"

The `function_composition.py` operation descriptions are also structure-aware, using "scalar multiply by k" for additive groups and "compose the current element with itself k times" for non-abelian groups.

**Files changed:** `skins.py`, `composers/function_composition.py`

### 2. Missing Information in Prompts (Defect #2)

**Problem:** For SymmetricGroup and DihedralGroup, the `power_k` operation had no explicit handler in the skin's `op_description()` method. It fell through to generic fallback phrases like "apply a shuffle" (DeckOfCards), "rearrange the seats" (Seating), "apply a symmetry" (PolygonSymmetries), or "apply a tile move" (TileFlip). These phrases provide no information about what operation to perform, making the task unsolvable.

**Fix:** All 4 non-abelian skins now have explicit `"power"` handlers:
- DeckOfCardsSkin: "compose the current arrangement with itself k times (apply it as a shuffle k times)"
- SeatingSkin: "apply the current seating rearrangement k times in succession"
- PolygonSymmetriesSkin: "compose the current transformation with itself k times"
- TileFlipSkin: "apply the current tile orientation k times in succession"

Additionally, conjugation operations now have explicit handlers in all skins.

**Files changed:** `skins.py`

### 3. Element Labeling Off-by-One (Defect #3)

**Problem:** S_n elements are 1-indexed tuples (e.g., S_3 identity = (1, 2, 3)). The old DeckOfCardsSkin used a 0-indexed mapping: `{0: "Ace", 1: "Two", 2: "Three", 3: "Four", ...}`. So S_3 elements (containing values 1, 2, 3) got mapped to "Two, Three, Four" — confusing and inconsistent with the standard S_3 = {1, 2, 3}.

**Fix:**
- DeckOfCardsSkin now uses simple numbered labels: `Card 1, Card 2, Card 3, ...` which directly match the 1-indexed element values.
- SeatingSkin now uses a `_person_name(i)` method that converts 1-indexed values: `1→Alice, 2→Bob, 3→Carol, ...`

**Files changed:** `skins.py`

---

## New Features

### Built-in Task Validation Layer

A new `TaskValidator` class and `algebraid validate` CLI command provide post-generation quality assurance:

- **Prompt completeness check**: Detects generic fallback phrases that indicate missing operation parameters
- **Answer consistency check**: Verifies that solution traces match answer_raw values
- **Structural integrity check**: Validates required fields, task_id uniqueness, and depth values
- **Template variable detection**: Catches unresolved `{variable}` placeholders in prompts
- **Formatting check**: Warns about double periods and other formatting issues

The validation is automatically run after `algebraid generate` (can be skipped with `--skip-validation`).

**Files added:** `tasks/validator.py`
**Files changed:** `cli.py`, `tasks/__init__.py`, `__init__.py`

---

## Files Changed

| File | Change |
|---|---|
| `src/algebraid/skins.py` | Fixed all 10 skins: power descriptions, element labeling, conjugation handlers |
| `src/algebraid/composers/function_composition.py` | Structure-aware power operation descriptions |
| `src/algebraid/tasks/validator.py` | NEW: Task validation module |
| `src/algebraid/tasks/__init__.py` | Added validator exports |
| `src/algebraid/__init__.py` | Added validator exports, bumped to v2.1.0 |
| `src/algebraid/cli.py` | Added `validate` subcommand, auto-validation in `generate` |
| `README.md` | Comprehensive rewrite with v2.1 changes, operation semantics table |
| `pyproject.toml` | Version bump to 2.1.0 |
| `test_fixes.py` | NEW: Automated test suite for all defect fixes |
| `verify_samples.py` | NEW: Manual verification script with sample prompts |
| `results/tasks_v2.1.jsonl` | NEW: Clean validated task set (312 tasks, 100% pass rate) |

---

## Test Results

All 5 test suites pass:

1. **Power operation semantics**: Z_5, GF(7), S_3 all produce correct results with accurate descriptions
2. **Element labeling consistency**: S_3 identity correctly labeled as [Card 1, Card 2, Card 3] and [Alice, Bob, Carol]
3. **All skins handle power operations**: 10/10 skins produce specific (non-generic) power descriptions
4. **Feedback defect reproduction**: All 3 specific scenarios from the Gemini review now produce correct results
5. **Full generation + validation**: 312 tasks generated, 100% validation pass rate, 0 errors, 0 warnings
