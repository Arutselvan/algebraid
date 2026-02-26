"""
Tests for AlgebraidEvaluator and EvalReport.

Coverage:
  - Correct/incorrect scoring
  - accuracy_by_depth, accuracy_by_family, accuracy_by_dimension breakdowns
  - Compositional ceiling computed on chain families only
  - Missing predictions counted, not crashed on
  - EvalReport.to_dict / from_dict round-trip
  - print_summary does not raise
"""

import json
import pytest

from algebraid.evaluator import AlgebraidEvaluator, EvalReport, EvalResult
from algebraid.task_model import Task, TaskSet, TaskFamily, CompositionDimension


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _task(task_id, answer, family=TaskFamily.INTRA_STRUCTURE,
          dimension=CompositionDimension.GENERAL, depth=1,
          structures=None, solution_trace=None):
    return Task(
        task_id=task_id,
        prompt="Compute something.",
        answer=answer,
        answer_raw=answer,
        depth=depth,
        family=family,
        dimension=dimension,
        structures=structures or ["Z_7"],
        solution_trace=solution_trace,
    )


def _task_set(tasks):
    return TaskSet(tasks=tasks, name="test_set", description="test")


@pytest.fixture
def evaluator():
    return AlgebraidEvaluator()


@pytest.fixture
def mixed_task_set():
    """A task set with chain, conceptual, adversarial, and intermediate tasks."""
    return _task_set([
        _task("t-intra-1", "3", TaskFamily.INTRA_STRUCTURE, depth=1),
        _task("t-intra-2", "5", TaskFamily.INTRA_STRUCTURE, depth=2),
        _task("t-inter-1", "(2, 1)", TaskFamily.INTER_STRUCTURE, depth=1,
              structures=["Z_3", "Z_5"]),
        _task("t-field-1", "4", TaskFamily.FIELD_ARITHMETIC, depth=2,
              structures=["GF(7)"]),
        _task("t-concept-1", "0", TaskFamily.CONCEPTUAL_QUERY, depth=1),
        _task("t-concept-2", "yes", TaskFamily.CONCEPTUAL_QUERY, depth=1),
        _task("t-adv-1", "3", TaskFamily.INTRA_STRUCTURE,
              CompositionDimension.ADVERSARIAL, depth=2),
        _task("t-inter-2", "5", TaskFamily.INTERMEDIATE_STATE
              if hasattr(TaskFamily, "INTERMEDIATE_STATE")
              else TaskFamily.INTRA_STRUCTURE,
              CompositionDimension.INTERMEDIATE_STATE, depth=1),
    ])


# ── Basic scoring ──────────────────────────────────────────────────────────────

class TestBasicScoring:
    def test_all_correct(self, evaluator):
        ts = _task_set([_task("t1", "3"), _task("t2", "5")])
        preds = {"t1": "3", "t2": "5"}
        report = evaluator.evaluate(ts, preds, model_name="test")
        assert report.accuracy_overall == 1.0
        assert report.total_correct == 2

    def test_all_wrong(self, evaluator):
        ts = _task_set([_task("t1", "3"), _task("t2", "5")])
        preds = {"t1": "9", "t2": "9"}
        report = evaluator.evaluate(ts, preds, model_name="test")
        assert report.accuracy_overall == 0.0
        assert report.total_correct == 0

    def test_partial_correct(self, evaluator):
        ts = _task_set([_task("t1", "3"), _task("t2", "5"), _task("t3", "1")])
        preds = {"t1": "3", "t2": "9", "t3": "1"}
        report = evaluator.evaluate(ts, preds, model_name="test")
        assert report.total_correct == 2
        assert pytest.approx(report.accuracy_overall, abs=0.01) == 2 / 3

    def test_missing_prediction_scored_wrong(self, evaluator):
        ts = _task_set([_task("t1", "3"), _task("t2", "5")])
        preds = {"t1": "3"}   # t2 missing
        report = evaluator.evaluate(ts, preds, model_name="test")
        assert report.missing_predictions == 1
        assert report.total_correct == 1

    def test_empty_predictions_dict(self, evaluator):
        ts = _task_set([_task("t1", "3")])
        report = evaluator.evaluate(ts, {}, model_name="test")
        assert report.accuracy_overall == 0.0
        assert report.missing_predictions == 1


# ── Accuracy breakdowns ────────────────────────────────────────────────────────

class TestAccuracyBreakdowns:
    def test_by_depth_keys(self, evaluator):
        ts = _task_set([
            _task("t1", "3", depth=1),
            _task("t2", "5", depth=2),
            _task("t3", "1", depth=2),
        ])
        preds = {"t1": "3", "t2": "5", "t3": "9"}
        report = evaluator.evaluate(ts, preds, model_name="test")
        assert 1 in report.accuracy_by_depth
        assert 2 in report.accuracy_by_depth
        assert report.accuracy_by_depth[1]["accuracy"] == 1.0
        assert report.accuracy_by_depth[2]["accuracy"] == 0.5

    def test_by_family_keys(self, evaluator, mixed_task_set):
        preds = {t.task_id: t.answer for t in mixed_task_set}
        report = evaluator.evaluate(mixed_task_set, preds, model_name="test")
        assert "intra-structure composition" in report.accuracy_by_family
        assert "inter-structure composition" in report.accuracy_by_family
        assert "conceptual query" in report.accuracy_by_family

    def test_adversarial_under_intra_family(self, evaluator):
        """Adversarial tasks use INTRA_STRUCTURE family label."""
        ts = _task_set([
            _task("t-adv", "3", TaskFamily.INTRA_STRUCTURE,
                  CompositionDimension.ADVERSARIAL, depth=2),
        ])
        preds = {"t-adv": "3"}
        report = evaluator.evaluate(ts, preds, model_name="test")
        assert "intra-structure composition" in report.accuracy_by_family
        assert "adversarial" not in report.accuracy_by_family

    def test_adversarial_under_adversarial_dimension(self, evaluator):
        ts = _task_set([
            _task("t-adv", "3", TaskFamily.INTRA_STRUCTURE,
                  CompositionDimension.ADVERSARIAL, depth=2),
        ])
        preds = {"t-adv": "3"}
        report = evaluator.evaluate(ts, preds, model_name="test")
        assert "adversarial" in report.accuracy_by_dimension

    def test_by_dimension_keys(self, evaluator, mixed_task_set):
        preds = {t.task_id: t.answer for t in mixed_task_set}
        report = evaluator.evaluate(mixed_task_set, preds, model_name="test")
        assert "general" in report.accuracy_by_dimension
        assert "adversarial" in report.accuracy_by_dimension
        assert "intermediate_state" in report.accuracy_by_dimension


# ── Compositional ceiling ──────────────────────────────────────────────────────

class TestCompositionalCeiling:
    def test_ceiling_uses_chain_families_only(self, evaluator):
        """Conceptual tasks at depth=1 with 100% accuracy should not inflate
        the ceiling if chain tasks at depth=1 have lower accuracy."""
        ts = _task_set([
            # Chain task at depth=1: WRONG
            _task("intra-d1", "3", TaskFamily.INTRA_STRUCTURE, depth=1),
            # Chain task at depth=2: CORRECT
            _task("intra-d2", "5", TaskFamily.INTRA_STRUCTURE, depth=2),
            # Conceptual task at depth=1: CORRECT (easy)
            _task("concept-d1", "0", TaskFamily.CONCEPTUAL_QUERY, depth=1),
        ])
        preds = {"intra-d1": "9", "intra-d2": "5", "concept-d1": "0"}
        report = evaluator.evaluate(ts, preds, model_name="test")
        # depth-1 accuracy for chain tasks = 0%, so ceiling_50 should be depth=2
        # (not inflated to depth=1 by the conceptual task's correct answer)
        assert report.compositional_ceiling_50 == 2

    def test_ceiling_none_when_all_below_threshold(self, evaluator):
        ts = _task_set([_task("t1", "3", depth=1), _task("t2", "5", depth=2)])
        preds = {"t1": "9", "t2": "9"}
        report = evaluator.evaluate(ts, preds, model_name="test")
        assert report.compositional_ceiling_50 is None
        assert report.compositional_ceiling_25 is None


# ── Serialization ──────────────────────────────────────────────────────────────

class TestSerialization:
    def test_to_dict_is_json_serializable(self, evaluator):
        ts = _task_set([_task("t1", "3")])
        report = evaluator.evaluate(ts, {"t1": "3"}, model_name="test")
        d = report.to_dict()
        assert json.dumps(d)  # should not raise

    def test_from_dict_round_trip(self, evaluator):
        ts = _task_set([_task("t1", "3"), _task("t2", "5")])
        report = evaluator.evaluate(ts, {"t1": "3", "t2": "9"}, model_name="mymodel")
        d = report.to_dict()
        restored = EvalReport.from_dict(d)
        assert restored.model_name == "mymodel"
        assert restored.total_correct == 1
        assert restored.accuracy_overall == pytest.approx(0.5)
        assert len(restored.results) == 2

    def test_from_dict_preserves_per_task_results(self, evaluator):
        ts = _task_set([_task("t1", "3")])
        report = evaluator.evaluate(ts, {"t1": "3"}, model_name="test")
        restored = EvalReport.from_dict(report.to_dict())
        assert restored.results[0].task_id == "t1"
        assert restored.results[0].correct is True


# ── print_summary ──────────────────────────────────────────────────────────────

class TestPrintSummary:
    def test_does_not_raise(self, evaluator, capsys):
        ts = _task_set([_task("t1", "3"), _task("t2", "5")])
        report = evaluator.evaluate(ts, {"t1": "3", "t2": "9"}, model_name="test")
        report.print_summary()
        out = capsys.readouterr().out
        assert "50.0%" in out or "1/2" in out
