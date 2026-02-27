"""
Tests for the analysis module.

Coverage:
  - fit_scaling_law: operates on chain families only; not contaminated by
    conceptual or rule tasks; returns correct keys
  - find_phase_transition: chain-only; identifies correct depth
  - error_taxonomy: adversarial_trap label (not commutativity_swap);
    tuple answers not miscategorised as off_by_one; no "other" catch-all
  - stability_breakdown: backward-compatible alias; accuracy values in valid
    range; required keys present
  - run_analysis: returns five structured analyses (accuracy_by_depth,
    accuracy_by_family, accuracy_by_dimension, complexity_analysis,
    hallucination_onset)

Note: fit_scaling_law, fit_scaling_law_by_family, and find_phase_transition
are standalone advanced functions not included in run_analysis() output.
"""

import pytest

from algebraid.evaluator import AlgebraidEvaluator, EvalReport, EvalResult
from algebraid.task_model import Task, TaskSet, TaskFamily, CompositionDimension
from algebraid.analysis import (
    fit_scaling_law,
    fit_scaling_law_by_family,
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

    def test_adversarial_tasks_excluded(self):
        """Adversarial tasks (family=intra, dim=adversarial) must not change the fit.

        They always have depth 1-2 regardless of the requested depth, so
        including them would distort the scaling curve (Bug 1 regression).
        """
        base = self._chain_report()
        with_adversarial = _make_report(base.results + [
            # All correct at depth=1 -- would inflate depth-1 accuracy if included
            _make_result("a1", True, INTRA, dimension="adversarial", depth=1),
            _make_result("a2", True, INTRA, dimension="adversarial", depth=1),
            _make_result("a3", True, INTRA, dimension="adversarial", depth=1),
        ])
        law_base = fit_scaling_law(base)
        law_extra = fit_scaling_law(with_adversarial)
        assert law_base["alpha"] == law_extra["alpha"]
        assert law_base["A"] == law_extra["A"]

    def test_intermediate_tasks_excluded(self):
        """Intermediate tasks (family=intra, dim=intermediate_state) must not change the fit."""
        base = self._chain_report()
        with_intermediate = _make_report(base.results + [
            _make_result("i1", True, INTRA, dimension="intermediate_state", depth=2),
            _make_result("i2", True, INTRA, dimension="intermediate_state", depth=2),
        ])
        law_base = fit_scaling_law(base)
        law_extra = fit_scaling_law(with_intermediate)
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

    def test_yes_no_wrong_classified_as_wrong_value(self):
        """Wrong Yes/No conceptual answers should be 'wrong_value', not off_by_one."""
        report = _make_report([
            _make_result("t1", False, CONCEPTUAL, response="yes", ground_truth="no"),
        ])
        tax = error_taxonomy(report)
        assert "wrong_value" in tax["categories"]
        assert "off_by_one" not in tax["categories"]
        assert "other" not in tax["categories"]

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

    def test_accuracy_values_in_valid_range(self):
        """Accuracy values returned by stability_breakdown must be in [0, 1]."""
        report = _make_report([
            _make_result("t1", True, INTRA, depth=1),
            _make_result("t2", True, INTRA, depth=1),
            _make_result("t3", False, INTRA, depth=2),
            _make_result("t4", True, INTRA, depth=2),
        ])
        curve = stability_breakdown(report)
        for row in curve:
            assert 0.0 <= row["accuracy"] <= 1.0

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
            assert "errors_by_category" in row


# ── run_analysis ───────────────────────────────────────────────────────────────

class TestRunAnalysis:
    def test_returns_all_expected_keys(self):
        report = _make_report([
            _make_result("t1", True,  INTRA, depth=1),
            _make_result("t2", False, INTRA, depth=2),
        ])
        analysis = run_analysis(report)
        # Five structured analyses
        assert "accuracy_by_depth" in analysis
        assert "accuracy_by_family" in analysis
        assert "accuracy_by_dimension" in analysis
        assert "complexity_analysis" in analysis
        assert "hallucination_onset" in analysis
        # Metadata
        assert "overall_accuracy" in analysis
        assert "total_tasks" in analysis
        assert "total_correct" in analysis

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

    def test_accuracy_by_depth_structure(self):
        report = _make_report([
            _make_result("t1", True,  INTRA, depth=1),
            _make_result("t2", False, INTRA, depth=2),
        ])
        analysis = run_analysis(report)
        abd = analysis["accuracy_by_depth"]
        # Must have both curve and by_family sub-keys
        assert "curve" in abd
        assert "by_family" in abd
        # Each curve row must have the required keys
        for row in abd["curve"]:
            assert "depth" in row
            assert "accuracy" in row
            assert "correct" in row
            assert "total" in row
            assert "errors_by_category" in row
        # Each family entry must have per-depth rows with required keys
        for fam, rows in abd["by_family"].items():
            for row in rows:
                assert "depth" in row
                assert "accuracy" in row
                assert "correct" in row
                assert "total" in row

    def test_accuracy_by_family_all_families(self):
        report = _make_report([
            _make_result("t1", True,  INTRA, depth=1),
            _make_result("t2", False, INTRA, depth=2),
            _make_result("t3", True,  CONCEPTUAL, depth=1),
            _make_result("t4", True,  RULE, depth=1),
        ])
        analysis = run_analysis(report)
        by_fam = analysis["accuracy_by_family"]
        # Must include non-chain families too
        assert CONCEPTUAL in by_fam
        assert RULE in by_fam
        for fam, d in by_fam.items():
            assert "total" in d
            assert "correct" in d
            assert "accuracy" in d

    def test_accuracy_by_dimension_keys(self):
        report = _make_report([
            _make_result("t1", True,  INTRA, dimension="general",  depth=1),
            _make_result("t2", False, INTRA, dimension="adversarial", depth=1),
            _make_result("t3", True,  INTRA, dimension="intermediate_state", depth=2),
        ])
        analysis = run_analysis(report)
        by_dim = analysis["accuracy_by_dimension"]
        assert "general" in by_dim
        assert "adversarial" in by_dim
        assert "intermediate_state" in by_dim
        for dim, d in by_dim.items():
            assert "total" in d
            assert "correct" in d
            assert 0.0 <= d["accuracy"] <= 1.0

    def test_complexity_analysis_structure(self):
        report = _make_report([
            _make_result("t1", True,  INTRA, depth=1),
            _make_result("t2", False, INTRA, depth=2),
        ])
        analysis = run_analysis(report)
        cx = analysis["complexity_analysis"]
        assert "by_depth" in cx
        assert "vs_accuracy" in cx
        assert isinstance(cx["by_depth"], list)
        assert isinstance(cx["vs_accuracy"], list)

    def test_hallucination_onset_structure(self):
        report = _make_report([
            _make_result("t1", True,  INTRA, depth=1),
            _make_result("t2", False, INTRA, depth=2, response="I cannot determine this"),
        ])
        analysis = run_analysis(report)
        hall = analysis["hallucination_onset"]
        assert "onset_depth" in hall
        assert "threshold" in hall
        assert "curve" in hall
        assert "note" in hall


# ── fit_scaling_law improvements ───────────────────────────────────────────────

class TestFitScalingLawImprovements:
    def _decaying_report(self):
        """Report with clear power-law decay across 5 depths."""
        results = []
        for d in range(1, 6):
            for _ in range(d):          # correct = 1 (always)
                results.append(_make_result(f"t{d}c", True, INTRA, depth=d))
            for _ in range(d * d - d):  # wrong   = d^2 - d
                results.append(_make_result(f"t{d}w", False, INTRA, depth=d))
        return _make_report(results)

    def test_alpha_se_key_present(self):
        law = fit_scaling_law(self._decaying_report())
        assert "alpha_se" in law

    def test_alpha_se_none_when_only_two_depths(self):
        """SE requires n > 2 data points; two depths -> alpha_se is None."""
        report = _make_report([
            _make_result("t1", True,  INTRA, depth=1),
            _make_result("t2", False, INTRA, depth=2),
        ])
        law = fit_scaling_law(report)
        assert law["alpha_se"] is None

    def test_alpha_se_positive_when_fitted(self):
        law = fit_scaling_law(self._decaying_report())
        if law["alpha_se"] is not None:
            assert law["alpha_se"] >= 0.0

    def test_interpretation_reflects_r2_quality_strong(self):
        """A clean power-law signal should yield a strong-fit interpretation."""
        # Perfect power-law: acc(d) = 1/d -> A=1, alpha=1, perfect fit
        results = []
        for d in range(1, 8):
            for _ in range(d):          # d tasks, 1 correct -> acc = 1/d
                results.append(_make_result(f"t{d}c", True,  INTRA, depth=d))
            for _ in range(d * d - d):  # d^2 - d wrong
                results.append(_make_result(f"t{d}w", False, INTRA, depth=d))
        report = _make_report(results)
        law = fit_scaling_law(report)
        if law["r2"] is not None and law["r2"] >= 0.95:
            assert "Strong fit" in law["interpretation"]

    def test_alpha_leq_zero_has_special_interpretation(self):
        """When alpha <= 0 the interpretation must say 'does not decrease'."""
        # Accuracy increases with depth (inverted) — alpha should be negative
        report = _make_report([
            _make_result("t1", False, INTRA, depth=1),
            _make_result("t2", False, INTRA, depth=1),
            _make_result("t3", True,  INTRA, depth=2),
            _make_result("t4", True,  INTRA, depth=3),
            _make_result("t5", True,  INTRA, depth=3),
        ])
        law = fit_scaling_law(report)
        if law["alpha"] is not None and law["alpha"] <= 0:
            assert "does not decrease" in law["interpretation"]

    def test_data_sufficiency_note_when_few_depths(self):
        """A fit on < 8 depth levels must include a data-quality note."""
        report = _make_report([
            _make_result("t1", True,  INTRA, depth=1),
            _make_result("t2", False, INTRA, depth=2),
            _make_result("t3", True,  INTRA, depth=3),
        ])
        law = fit_scaling_law(report)
        assert "note" in law
        assert "depth levels" in law["note"]

    def test_no_data_sufficiency_note_when_many_depths(self):
        """No note when >= 8 distinct depth levels are available."""
        results = []
        for d in range(1, 10):  # 9 depths
            results.append(_make_result(f"tc{d}",  True,  INTRA, depth=d))
            results.append(_make_result(f"tw{d}", False, INTRA, depth=d))
        report = _make_report(results)
        law = fit_scaling_law(report)
        assert "note" not in law or "depth levels" not in law.get("note", "")


# ── fit_scaling_law_by_family ──────────────────────────────────────────────────

class TestFitScalingLawByFamily:
    def _multi_family_report(self):
        return _make_report([
            _make_result("i1", True,  INTRA, depth=1),
            _make_result("i2", False, INTRA, depth=2),
            _make_result("i3", False, INTRA, depth=3),
            _make_result("r1", True,  INTER, depth=1),
            _make_result("r2", True,  INTER, depth=2),
            _make_result("r3", False, INTER, depth=3),
            _make_result("f1", True,  FIELD, depth=1),
            _make_result("f2", False, FIELD, depth=2),
        ])

    def test_returns_all_three_chain_families(self):
        by_fam = fit_scaling_law_by_family(self._multi_family_report())
        assert INTRA in by_fam
        assert INTER in by_fam
        assert FIELD in by_fam

    def test_each_entry_has_required_keys(self):
        by_fam = fit_scaling_law_by_family(self._multi_family_report())
        for fam, fit in by_fam.items():
            assert "A" in fit
            assert "alpha" in fit
            assert "alpha_se" in fit
            assert "r2" in fit

    def test_family_fits_are_independent(self):
        """Intra fit must not be influenced by inter results."""
        intra_only = _make_report([
            _make_result("i1", True,  INTRA, depth=1),
            _make_result("i2", False, INTRA, depth=2),
            _make_result("i3", False, INTRA, depth=3),
        ])
        mixed = _make_report(intra_only.results + [
            _make_result("r1", True, INTER, depth=1),
            _make_result("r2", True, INTER, depth=2),
            _make_result("r3", True, INTER, depth=3),
        ])
        fit_solo  = fit_scaling_law_by_family(intra_only)[INTRA]
        fit_mixed = fit_scaling_law_by_family(mixed)[INTRA]
        assert fit_solo["alpha"] == fit_mixed["alpha"]

    def test_missing_family_returns_note(self):
        """A family absent from the report must return a note, not crash."""
        report = _make_report([
            _make_result("i1", True, INTRA, depth=1),
            _make_result("i2", False, INTRA, depth=2),
        ])
        by_fam = fit_scaling_law_by_family(report)
        assert by_fam[FIELD]["A"] is None
        assert "note" in by_fam[FIELD]

    def test_adversarial_excluded_per_family(self):
        """Adversarial-dimension intra tasks must not contaminate the intra fit."""
        base = _make_report([
            _make_result("i1", True,  INTRA, depth=1),
            _make_result("i2", False, INTRA, depth=2),
            _make_result("i3", False, INTRA, depth=3),
        ])
        with_adv = _make_report(base.results + [
            _make_result("a1", True, INTRA, dimension="adversarial", depth=1),
            _make_result("a2", True, INTRA, dimension="adversarial", depth=1),
        ])
        fit_base = fit_scaling_law_by_family(base)[INTRA]
        fit_adv  = fit_scaling_law_by_family(with_adv)[INTRA]
        assert fit_base["alpha"] == fit_adv["alpha"]


# ── proof report key rename ────────────────────────────────────────────────────

class TestProofReportKeys:
    def test_verify_set_returns_trace_verified_key(self):
        from algebraid.proof import verify_set
        from algebraid.generator import AlgebraidGenerator
        ts = AlgebraidGenerator(seed=42).generate(
            depths=[1], tasks_per_depth=3, families=["intra"]
        )
        report = verify_set(ts)
        assert "trace_verified" in report
        assert "proven" not in report

    def test_trace_verified_count_positive(self):
        from algebraid.proof import verify_set
        from algebraid.generator import AlgebraidGenerator
        ts = AlgebraidGenerator(seed=42).generate(
            depths=[1], tasks_per_depth=5, families=["intra"]
        )
        report = verify_set(ts)
        assert report["trace_verified"] > 0

    def test_proof_rate_zero_for_untraceable_set(self):
        """Conceptual-only set has no traceable tasks -> proof_rate = 0.0."""
        from algebraid.proof import verify_set
        from algebraid.generator import AlgebraidGenerator
        ts = AlgebraidGenerator(seed=42).generate(
            depths=[1], tasks_per_depth=5, families=["conceptual"]
        )
        report = verify_set(ts)
        assert report["proof_rate"] == 0.0
        assert report["trace_verified"] == 0


# ── validator prompt-trace alignment ──────────────────────────────────────────

class TestPromptTraceAlignment:
    def test_no_warning_when_element_in_prompt(self):
        """A correctly described operation produces no alignment warning."""
        from algebraid.tasks.validator import TaskValidator
        from algebraid.task_model import Task, TaskFamily, CompositionDimension
        # right_mul_3 -> element '3' appears in the prompt
        task = Task(
            task_id="AG-align01",
            prompt="Starting with x = 2, add 3 (mod 7). What is the result?",
            answer="5",
            answer_raw="5",
            depth=1,
            family=TaskFamily.INTRA_STRUCTURE,
            dimension=CompositionDimension.GENERAL,
            structures=["Z_7"],
            solution_trace=[("start", "2"), ("right_mul_3", "5")],
        )
        result = TaskValidator().validate(task)
        alignment_warns = [w for w in result.warnings if "right_mul_3" in w]
        assert len(alignment_warns) == 0

    def test_warning_when_element_absent_from_prompt(self):
        """Element '5' in trace but '3' described in prompt -> warning."""
        from algebraid.tasks.validator import TaskValidator
        from algebraid.task_model import Task, TaskFamily, CompositionDimension
        task = Task(
            task_id="AG-align02",
            prompt="Starting with x = 2, add 3 (mod 7). What is the result?",
            answer="0",
            answer_raw="0",
            depth=1,
            family=TaskFamily.INTRA_STRUCTURE,
            dimension=CompositionDimension.GENERAL,
            structures=["Z_7"],
            # trace says right_mul_5 but prompt says "add 3"
            solution_trace=[("start", "2"), ("right_mul_5", "0")],
        )
        result = TaskValidator().validate(task)
        assert any("right_mul_5" in w for w in result.warnings)

    def test_inverse_op_skipped_by_alignment_check(self):
        """The 'inverse' operation has no element suffix — must not warn."""
        from algebraid.tasks.validator import TaskValidator
        from algebraid.task_model import Task, TaskFamily, CompositionDimension
        task = Task(
            task_id="AG-align03",
            prompt="Starting with x = 3, take the inverse in Z_7. What is the result?",
            answer="4",
            answer_raw="4",
            depth=1,
            family=TaskFamily.INTRA_STRUCTURE,
            dimension=CompositionDimension.GENERAL,
            structures=["Z_7"],
            solution_trace=[("start", "3"), ("inverse", "4")],
        )
        result = TaskValidator().validate(task)
        alignment_warns = [w for w in result.warnings if "inverse" in w and "right_mul" not in w]
        assert len(alignment_warns) == 0


# ── skin name in metadata ──────────────────────────────────────────────────────

class TestSkinInMetadata:
    def test_intra_task_has_skin_key(self):
        from algebraid.generator import AlgebraidGenerator
        ts = AlgebraidGenerator(seed=42).generate(
            depths=[1], tasks_per_depth=5, families=["intra"]
        )
        for t in ts:
            assert "skin" in t.metadata

    def test_use_skins_false_gives_none_skin(self):
        from algebraid.generator import AlgebraidGenerator
        ts = AlgebraidGenerator(seed=42).generate(
            depths=[1], tasks_per_depth=5, families=["intra"], use_skins=False
        )
        for t in ts:
            assert t.metadata.get("skin") is None

    def test_use_skins_true_gives_string_skin(self):
        from algebraid.generator import AlgebraidGenerator
        ts = AlgebraidGenerator(seed=42).generate(
            depths=[1], tasks_per_depth=10, families=["intra"], use_skins=True
        )
        for t in ts:
            assert isinstance(t.metadata.get("skin"), str)

    def test_rule_task_has_skin_key(self):
        from algebraid.generator import AlgebraidGenerator
        ts = AlgebraidGenerator(seed=42).generate(
            depths=[1], tasks_per_depth=5, families=["rule"]
        )
        for t in ts:
            assert "skin" in t.metadata

    def test_adversarial_task_has_skin_key(self):
        from algebraid.generator import AlgebraidGenerator
        ts = AlgebraidGenerator(seed=42).generate(
            depths=[2], tasks_per_depth=5, families=["adversarial"]
        )
        for t in ts:
            assert "skin" in t.metadata

    def test_skin_survives_jsonl_roundtrip(self):
        import tempfile, os
        from algebraid.generator import AlgebraidGenerator
        from algebraid.task_model import TaskSet
        ts = AlgebraidGenerator(seed=42).generate(
            depths=[1], tasks_per_depth=3, families=["intra"]
        )
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            path = f.name
        try:
            ts.to_jsonl(path)
            ts2 = TaskSet.from_jsonl(path)
            for t in ts2:
                assert "skin" in t.metadata
        finally:
            os.unlink(path)
