"""
Tests for the splits module.

Coverage:
  - split_by_depth: correct depth boundaries and gap
  - split_by_commutativity: abelian vs non-abelian
  - split_by_structure: prefix matching
  - split_by_family: key expansion, adversarial/intermediate opt-in (Bug 3)
  - split_summary: returns a non-empty string
"""

import pytest

from algebraid.task_model import Task, TaskSet, TaskFamily, CompositionDimension
from algebraid.splits import (
    split_by_depth,
    split_by_commutativity,
    split_by_structure,
    split_by_family,
    split_summary,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _task(task_id, family, dimension="general", depth=1, structures=None):
    return Task(
        task_id=task_id,
        prompt="p",
        answer="0",
        answer_raw="0",
        depth=depth,
        family=family,
        dimension=CompositionDimension(dimension),
        structures=structures or ["Z_7"],
    )


def _ts(tasks):
    return TaskSet(tasks=tasks, name="test")


INTRA = TaskFamily.INTRA_STRUCTURE
INTER = TaskFamily.INTER_STRUCTURE
FIELD = TaskFamily.FIELD_ARITHMETIC
RULE = TaskFamily.RULE_INDUCTION
CONCEPT = TaskFamily.CONCEPTUAL_QUERY


# ── split_by_depth ─────────────────────────────────────────────────────────────

class TestSplitByDepth:
    def _ts(self):
        return _ts([
            _task("d1", INTRA, depth=1),
            _task("d2", INTRA, depth=2),
            _task("d3", INTRA, depth=3),
            _task("d4", INTRA, depth=4),
        ])

    def test_train_contains_shallow(self):
        train, _ = split_by_depth(self._ts(), train_max_depth=2)
        ids = {t.task_id for t in train}
        assert "d1" in ids and "d2" in ids

    def test_train_excludes_deep(self):
        train, _ = split_by_depth(self._ts(), train_max_depth=2)
        ids = {t.task_id for t in train}
        assert "d4" not in ids

    def test_test_respects_default_gap(self):
        # Default gap=2: test >= train_max_depth + 2
        _, test = split_by_depth(self._ts(), train_max_depth=1)
        depths = {t.depth for t in test}
        assert all(d >= 3 for d in depths)

    def test_gap_task_excluded_from_both(self):
        # depth=2 is between train<=1 and test>=3 -> excluded from both
        train, test = split_by_depth(self._ts(), train_max_depth=1, test_min_depth=3)
        all_ids = {t.task_id for t in train} | {t.task_id for t in test}
        assert "d2" not in all_ids

    def test_custom_test_min_depth(self):
        _, test = split_by_depth(self._ts(), train_max_depth=1, test_min_depth=2)
        depths = {t.depth for t in test}
        assert min(depths) == 2


# ── split_by_commutativity ─────────────────────────────────────────────────────

class TestSplitByCommutativity:
    def _ts(self):
        return _ts([
            _task("z7",  INTRA, structures=["Z_7"]),
            _task("gf5", INTRA, structures=["GF(5)"]),
            _task("s3",  INTRA, structures=["S_3"]),
            _task("d4",  INTRA, structures=["D_4"]),
            _task("q8",  INTRA, structures=["Q_8"]),
        ])

    def test_abelian_in_train(self):
        train, _ = split_by_commutativity(self._ts())
        ids = {t.task_id for t in train}
        assert "z7" in ids and "gf5" in ids

    def test_nonabelian_in_test(self):
        _, test = split_by_commutativity(self._ts())
        ids = {t.task_id for t in test}
        assert "s3" in ids and "d4" in ids and "q8" in ids

    def test_abelian_not_in_test(self):
        _, test = split_by_commutativity(self._ts())
        ids = {t.task_id for t in test}
        assert "z7" not in ids


# ── split_by_structure ─────────────────────────────────────────────────────────

class TestSplitByStructure:
    def _ts(self):
        return _ts([
            _task("z5",  INTRA, structures=["Z_5"]),
            _task("z7",  INTRA, structures=["Z_7"]),
            _task("z11", INTRA, structures=["Z_11"]),
            _task("s3",  INTRA, structures=["S_3"]),
        ])

    def test_prefix_match(self):
        train, test = split_by_structure(self._ts(), ["Z_5", "Z_7"], ["Z_11"])
        train_ids = {t.task_id for t in train}
        test_ids = {t.task_id for t in test}
        assert "z5" in train_ids and "z7" in train_ids
        assert "z11" in test_ids

    def test_unmatched_excluded(self):
        train, test = split_by_structure(self._ts(), ["Z_5"], ["Z_11"])
        all_ids = {t.task_id for t in train} | {t.task_id for t in test}
        assert "s3" not in all_ids
        assert "z7" not in all_ids


# ── split_by_family ────────────────────────────────────────────────────────────

class TestSplitByFamily:
    def _ts(self):
        return _ts([
            _task("intra_std",  INTRA, dimension="general"),
            _task("inter",      INTER, dimension="general"),
            _task("field",      FIELD, dimension="general"),
            _task("rule",       RULE,  dimension="general"),
            _task("conceptual", CONCEPT, dimension="general"),
            _task("adv",        INTRA, dimension="adversarial"),
            _task("intermediate", INTRA, dimension="intermediate_state"),
        ])

    def test_intra_key_matches_standard_intra(self):
        train, _ = split_by_family(self._ts(), ["intra"], [])
        ids = {t.task_id for t in train}
        assert "intra_std" in ids

    def test_intra_key_excludes_adversarial(self):
        """Bug 3 regression: 'intra' must not include adversarial-dimension tasks."""
        train, _ = split_by_family(self._ts(), ["intra"], [])
        ids = {t.task_id for t in train}
        assert "adv" not in ids

    def test_intra_key_excludes_intermediate(self):
        """Bug 3 regression: 'intra' must not include intermediate_state tasks."""
        train, _ = split_by_family(self._ts(), ["intra"], [])
        ids = {t.task_id for t in train}
        assert "intermediate" not in ids

    def test_adversarial_key_includes_adversarial_tasks(self):
        _, test = split_by_family(self._ts(), [], ["adversarial"])
        ids = {t.task_id for t in test}
        assert "adv" in ids

    def test_intermediate_key_includes_intermediate_tasks(self):
        _, test = split_by_family(self._ts(), [], ["intermediate"])
        ids = {t.task_id for t in test}
        assert "intermediate" in ids

    def test_no_overlap_in_intra_vs_adversarial_split(self):
        """Adversarial tasks must not appear in both halves of the split."""
        train, test = split_by_family(
            self._ts(),
            ["intra", "inter", "field", "rule", "conceptual"],
            ["adversarial"],
        )
        train_ids = {t.task_id for t in train}
        test_ids = {t.task_id for t in test}
        # adversarial task should only be in test
        assert "adv" not in train_ids
        assert "adv" in test_ids

    def test_conceptual_key(self):
        train, _ = split_by_family(self._ts(), ["conceptual"], [])
        ids = {t.task_id for t in train}
        assert "conceptual" in ids
        assert "intra_std" not in ids

    def test_all_standard_families(self):
        train, _ = split_by_family(
            self._ts(), ["intra", "inter", "field", "rule", "conceptual"], []
        )
        ids = {t.task_id for t in train}
        assert "intra_std" in ids
        assert "inter" in ids
        assert "field" in ids
        assert "rule" in ids
        assert "conceptual" in ids
        # adversarial and intermediate were not requested
        assert "adv" not in ids
        assert "intermediate" not in ids


# ── split_summary ──────────────────────────────────────────────────────────────

class TestSplitSummary:
    def test_returns_string(self):
        ts = _ts([_task("t1", INTRA, depth=1), _task("t2", INTRA, depth=2)])
        train, test = split_by_depth(ts, train_max_depth=1, test_min_depth=2)
        s = split_summary(train, test)
        assert isinstance(s, str)
        assert len(s) > 0
