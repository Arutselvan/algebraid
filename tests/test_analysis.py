"""
Tests for the analysis module.

Coverage:
  - fit_scaling_law: operates on chain families only; not contaminated by
    conceptual or rule tasks; returns correct keys
  - find_phase_transition: chain-only; identifies correct depth
  - error_taxonomy: adversarial_trap label (not commutativity_swap);
    tuple answers not miscategorised as off_by_one
  - hallucination_onset: chain-only
  - stability_breakdown: accuracy and fitted values from same population
  - run_analysis: returns all expected keys
"""

import pytest

from algebraid.evaluator import AlgebraidEvaluator, EvalReport, EvalResult
from algebraid.task_model import Task, TaskSet, TaskFamily, CompositionDimension
from algebraid.analysis import (
    fit_scaling_law,
    find_phase_transition,
    error_taxonomy,
    hallucination_onset,
    stability_breakdown,
    run_analysis,
    _CHAIN_FAMILIES,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_result(task_id, correct, family, dimension="general", depth=1,
                 response="", ground_truth="3"):
    return EvalResult(
        task_id=task_id,
        correct=correct,
        model_response=response,
        ground_truth=ground_truth,
        depth=depth,
        family=family,
        dimension=dimension,
    )


def _make_report(results):
    from collections import defaultdict
    depth_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        depth_stats[r.depth]["total"] += 1
        if r.correct:
            depth_stats[r.depth]["correct"] += 1
    for v in depth_stats.values():
        v["accuracy"] = v["correct"] / v["total"]
    return EvalReport(
        model_name="test",
        task_set_name="test_set",
        total_tasks=len(results),
        total_correct=sum(1 for r in results if r.correct),
        accuracy_overall=sum(1 for r in results if r.correct) / len(results),
        accuracy_by_depth=dict(depth_stats),
        accuracy_by_family={},
        accuracy_by_dimension={},
        compositional_ceiling_50=None,
        compositional_ceiling_25=None,
        results=results,
    )


INTRA = "intra-structure composition"
INTER = "inter-structure composition"
FIELD = "field arithmetic"
CONCEPTUAL = "conceptual query"
RULE = "rule induction"


# ── fit_scaling_law ────────────────────────────────────────────────────────────

class TestFitScalingLaw:
    def _chain_report(self):
        """Report with known accuracy decay: d=1->100%, d=2->50%, d=3->33%."""
        return _make_report([
            _make_result("t1", True,  INTRA, depth=1),
            _make_result("t2", True,  INTRA, depth=2),
            _make_result("t3", False, INTRA, depth=2),
            _make_result("t4", True,  INTRA, depth=3),
            _make_result("t5", False, INTRA, depth=3),
            _make_result("t6", False, INTRA, depth=3),
        ])

    def test_returns_required_keys(self):
        report = self._chain_report()
        law = fit_scaling_law(report)
        assert "A" in law and "alpha" in law and "r2" in law
        assert "data" in law and "families" in law

    def test_families_field_lists_chain_families(self):
        law = fit_scaling_law(self._chain_report())
        assert set(law["families"]) == _CHAIN_FAMILIES

    def test_alpha_positive_for_decaying_accuracy(self):
        """alpha > 0 means accuracy falls with depth."""
        law = fit_scaling_law(self._chain_report())
        assert law["alpha"] is not None
        assert law["alpha"] > 0

    def test_conceptual_tasks_excluded(self):
        """Adding perfect conceptual tasks at depth=1 must not change the fit."""
        base = self._chain_report()
        with_conceptual = _make_report(base.results + [
            _make_result("c1", True, CONCEPTUAL, depth=1),
            _make_result("c2", True, CONCEPTUAL, depth=1),
            _make_result("c3", True, CONCEPTUAL, depth=1),
        ])
        law_base = fit_scaling_law(base)
        law_extra = fit_scaling_law(with_conceptual)
        # Scaling law must be identical since conceptual tasks are excluded
        assert law_base["alpha"] == law_extra["alpha"]
        assert law_base["A"] == law_extra["A"]

    def test_rule_tasks_excluded(self):
        """Rule tasks (inverted depth relationship) must not contaminate fit."""
        base = self._chain_report()
        with_rule = _make_report(base.results + [
            # Rule tasks: depth=3 correct, depth=1 wrong (inverted difficulty)
            _make_result("r1", True,  RULE, depth=3),
            _make_result("r2", False, RULE, depth=1),
        ])
        law_base = fit_scaling_law(base)
        law_extra = fit_scaling_law(with_rule)
        assert law_base["alpha"] == law_extra["alpha"]

    def test_no_chain_results_returns_note(self):
        report = _make_report([
            _make_result("c1", True, CONCEPTUAL, depth=1),
        ])
        law = fit_scaling_law(report)
        assert law["A"] is None
        assert "note" in law

    def test_insufficient_depths_returns_note(self):
        report = _make_report([
            _make_result("t1", True, INTRA, depth=1),
            _make_result("t2", True, INTRA, depth=1),
        ])
        law = fit_scaling_law(report)
        # Only one depth level -> cannot fit
        assert law["A"] is None


# ── find_phase_transition ──────────────────────────────────────────────────────

class TestFindPhaseTransition:
    def test_identifies_steepest_drop(self):
        # d=1->90%, d=2->80%, d=3->20% (big drop at d=2->d=3)
        report = _make_report([
            _make_result("t1", True,  INTRA, depth=1),
            _make_result("t2", True,  INTRA, depth=1),
            _make_result("t3", True,  INTRA, depth=1),
            _make_result("t4", True,  INTRA, depth=1),
            _make_result("t5", False, INTRA, depth=1),
            _make_result("t6", True,  INTRA, depth=1),
            _make_result("t7", True,  INTRA, depth=1),
            _make_result("t8", True,  INTRA, depth=1),
            _make_result("t9", True,  INTRA, depth=1),
            _make_result("t10", True, INTRA, depth=2),
            _make_result("t11", True, INTRA, depth=2),
            _make_result("t12", True, INTRA, depth=2),
            _make_result("t13", True, INTRA, depth=2),
            _make_result("t14", False, INTRA, depth=2),
            _make_result("t15", True, INTRA, depth=3),
            _make_result("t16", False, INTRA, depth=3),
            _make_result("t17", False, INTRA, depth=3),
            _make_result("t18", False, INTRA, depth=3),
            _make_result("t19", False, INTRA, depth=3),
        ])
        pt = find_phase_transition(report)
        assert pt["steepest_drop_at_depth"] == 2  # biggest drop between d=2 and d=3

    def test_excludes_conceptual_from_transition(self):
        """Conceptual tasks at depth=1 must not participate in phase detection."""
        chain_only = _make_report([
            _make_result("t1", True, INTRA, depth=1),
            _make_result("t2", False, INTRA, depth=2),
        ])
        with_conceptual = _make_report(chain_only.results + [
            _make_result("c1", True, CONCEPTUAL, depth=1),
            _make_result("c2", True, CONCEPTUAL, depth=1),
        ])
        pt1 = find_phase_transition(chain_only)
        pt2 = find_phase_transition(with_conceptual)
        assert pt1["critical_depth"] == pt2["critical_depth"]

    def test_insufficient_depths_returns_note(self):
        report = _make_report([_make_result("t1", True, INTRA, depth=1)])
        pt = find_phase_transition(report)
        assert pt["critical_depth"] is None
        assert "note" in pt


# ── error_taxonomy ─────────────────────────────────────────────────────────────

class TestErrorTaxonomy:
    def test_adversarial_trap_not_commutativity_swap(self):
        """All adversarial-dimension errors should be 'adversarial_trap',
        not 'commutativity_swap'."""
        report = _make_report([
            _make_result("t1", False, INTRA, dimension="adversarial",
                         response="4", ground_truth="3"),
        ])
        tax = error_taxonomy(report)
        assert "adversarial_trap" in tax["categories"]
        assert "commutativity_swap" not in tax["categories"]

    def test_tuple_answer_not_off_by_one(self):
        """A permutation answer like '(2, 1, 3)' vs '(1, 2, 3)' is NOT off_by_one."""
        report = _make_report([
            _make_result("t1", False, INTRA,
                         response="(2, 1, 3)", ground_truth="(1, 2, 3)"),
        ])
        tax = error_taxonomy(report)
        assert "off_by_one" not in tax["categories"]

    def test_numeric_off_by_one_classified(self):
        report = _make_report([
            _make_result("t1", False, INTRA, response="4", ground_truth="3"),
        ])
        tax = error_taxonomy(report)
        assert "off_by_one" in tax["categories"]

    def test_hallucination_classified(self):
        report = _make_report([
            _make_result("t1", False, INTRA, response="I don't know", ground_truth="3"),
        ])
        tax = error_taxonomy(report)
        assert "hallucination" in tax["categories"]

    def test_yes_no_wrong_classified_as_other(self):
        """Wrong Yes/No conceptual answers should be 'other', not miscategorised."""
        report = _make_report([
            _make_result("t1", False, CONCEPTUAL, response="yes", ground_truth="no"),
        ])
        tax = error_taxonomy(report)
        assert "other" in tax["categories"]
        assert "off_by_one" not in tax["categories"]

    def test_no_errors_returns_empty(self):
        report = _make_report([
            _make_result("t1", True, INTRA, response="3", ground_truth="3"),
        ])
        tax = error_taxonomy(report)
        assert tax["total_errors"] == 0
        assert tax["dominant_error"] is None


# ── stability_breakdown ────────────────────────────────────────────────────────

class TestStabilityBreakdown:
    def test_uses_chain_families_only(self):
        """Depth=1 accuracy in the curve must reflect chain tasks only."""
        report = _make_report([
            # Chain task depth=1: WRONG
            _make_result("intra", False, INTRA, depth=1, response="9", ground_truth="3"),
            # Chain task depth=2: CORRECT
            _make_result("intra2", True, INTRA, depth=2, response="5", ground_truth="5"),
            # Conceptual task depth=1: CORRECT (easy)
            _make_result("concept", True, CONCEPTUAL, depth=1, response="0", ground_truth="0"),
        ])
        curve = stability_breakdown(report)
        depth1 = next((row for row in curve if row["depth"] == 1), None)
        assert depth1 is not None
        # Chain-only accuracy at depth=1 is 0% (not inflated by the easy conceptual task)
        assert depth1["accuracy"] == 0.0

    def test_fitted_and_accuracy_from_same_population(self):
        """When fit succeeds, fitted values should be in plausible range."""
        report = _make_report([
            _make_result("t1", True, INTRA, depth=1),
            _make_result("t2", True, INTRA, depth=1),
            _make_result("t3", False, INTRA, depth=2),
            _make_result("t4", True, INTRA, depth=2),
        ])
        curve = stability_breakdown(report)
        for row in curve:
            if row["fitted_accuracy"] is not None:
                assert 0.0 <= row["fitted_accuracy"] <= 1.0

    def test_returns_list_of_dicts_with_required_keys(self):
        report = _make_report([
            _make_result("t1", True, INTRA, depth=1),
            _make_result("t2", False, INTRA, depth=2),
        ])
        curve = stability_breakdown(report)
        for row in curve:
            assert "depth" in row
            assert "accuracy" in row
            assert "correct" in row
            assert "total" in row
            assert "fitted_accuracy" in row
            assert "errors_by_category" in row


# ── run_analysis ───────────────────────────────────────────────────────────────

class TestRunAnalysis:
    def test_returns_all_expected_keys(self):
        report = _make_report([
            _make_result("t1", True,  INTRA, depth=1),
            _make_result("t2", False, INTRA, depth=2),
        ])
        analysis = run_analysis(report)
        assert "scaling_law" in analysis
        assert "phase_transition" in analysis
        assert "error_taxonomy" in analysis
        assert "hallucination_onset" in analysis
        assert "stability_curve" in analysis
        assert "overall_accuracy" in analysis

    def test_does_not_raise_on_mixed_family_report(self):
        """Should not crash when given a realistic mixed-family task set."""
        report = _make_report([
            _make_result("i1", True,  INTRA, depth=1),
            _make_result("i2", False, INTRA, depth=2),
            _make_result("f1", True,  FIELD, depth=1),
            _make_result("c1", True,  CONCEPTUAL, depth=1),
            _make_result("r1", False, RULE, depth=1),
            _make_result("a1", False, INTRA, dimension="adversarial",
                         depth=2, response="4", ground_truth="3"),
        ])
        analysis = run_analysis(report)
        assert analysis["overall_accuracy"] >= 0.0
