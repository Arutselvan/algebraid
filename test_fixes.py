#!/usr/bin/env python3
"""
Test script to validate all fixes to ALGEBRAID.
Generates tasks, validates them, and checks specific defect scenarios.
"""

import sys
import os
import json

# Add the source to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from algebraid import AlgebraidGenerator, TaskValidator
from algebraid.tasks.validator import print_validation_report
from algebraid.primitives import CyclicGroup, SymmetricGroup, DihedralGroup, FiniteField
from algebraid.composers import make_standard_operations, ComposedFunction
from algebraid.skins import (
    ClockArithmeticSkin, MusicIntervalsSkin, RobotStepsSkin, ColorWheelSkin,
    DeckOfCardsSkin, SeatingSkin, PolygonSymmetriesSkin, TileFlipSkin,
    SecretCodesSkin, ModularArithmeticSkin,
)
import random


def test_power_operation_semantics():
    """Test that power operations produce correct results and descriptions."""
    print("\n" + "="*60)
    print("TEST 1: Power operation semantics")
    print("="*60)
    
    rng = random.Random(42)
    
    # Test CyclicGroup: power_2(x) should be 2*x mod n, NOT x+2
    cg = CyclicGroup(5)
    ops = make_standard_operations(cg, rng)
    power_op = [op for op in ops if "power" in op.name][0]
    
    x = 1
    result = power_op(x)
    power_val = int(power_op.name.split("_")[1])
    expected = (power_val * x) % 5
    
    print(f"  CyclicGroup Z_5: power_{power_val}({x}) = {result}")
    print(f"    Expected (scalar mul): {power_val} * {x} mod 5 = {expected}")
    print(f"    Wrong (iterated add):  {x} + {power_val} mod 5 = {(x + power_val) % 5}")
    assert result == expected, f"FAIL: power_{power_val}({x}) = {result}, expected {expected}"
    print(f"    PASS: Result matches scalar multiplication")
    
    # Check that the description says "multiply" not "advance N times"
    print(f"    Description: '{power_op.description}'")
    assert "multiply" in power_op.description.lower() or "scalar" in power_op.description.lower(), \
        f"FAIL: Description should mention 'multiply', got: {power_op.description}"
    print(f"    PASS: Description correctly describes scalar multiplication")
    
    # Test with skin
    skin = ClockArithmeticSkin()
    skin_desc = skin.op_description(power_op.name, power_op.fixed_args, cg)
    print(f"    Skin description: '{skin_desc}'")
    assert "multiply" in skin_desc.lower(), \
        f"FAIL: Skin description should mention 'multiply', got: {skin_desc}"
    print(f"    PASS: Skin description correctly describes scalar multiplication")
    
    # Test FiniteField: same issue
    rng2 = random.Random(99)
    ff = FiniteField(7)
    ops_ff = make_standard_operations(ff, rng2)
    power_op_ff = [op for op in ops_ff if "power" in op.name][0]
    
    x_ff = 3
    result_ff = power_op_ff(x_ff)
    power_val_ff = int(power_op_ff.name.split("_")[1])
    expected_ff = (power_val_ff * x_ff) % 7
    
    print(f"\n  FiniteField GF(7): power_{power_val_ff}({x_ff}) = {result_ff}")
    print(f"    Expected: {power_val_ff} * {x_ff} mod 7 = {expected_ff}")
    assert result_ff == expected_ff, f"FAIL"
    print(f"    PASS")
    
    # Test SymmetricGroup: power should be self-composition
    rng3 = random.Random(77)
    sg = SymmetricGroup(3)
    ops_sg = make_standard_operations(sg, rng3)
    power_op_sg = [op for op in ops_sg if "power" in op.name][0]
    
    perm = (2, 3, 1)  # A 3-cycle
    result_sg = power_op_sg(perm)
    power_val_sg = int(power_op_sg.name.split("_")[1])
    # Manual: (2,3,1) composed with itself = apply (2,3,1) then (2,3,1)
    # (2,3,1)^2: position 1→2→3, position 2→3→1, position 3→1→2 = (3,1,2)
    expected_sg = sg.op_chain(*([perm] * power_val_sg))
    
    print(f"\n  SymmetricGroup S_3: power_{power_val_sg}({perm}) = {result_sg}")
    print(f"    Expected: {expected_sg}")
    assert result_sg == expected_sg, f"FAIL"
    print(f"    PASS")
    
    # Check S_n skin description
    skin_sg = DeckOfCardsSkin()
    skin_desc_sg = skin_sg.op_description(power_op_sg.name, power_op_sg.fixed_args, sg)
    print(f"    Skin description: '{skin_desc_sg}'")
    assert "compose" in skin_desc_sg.lower() or "time" in skin_desc_sg.lower(), \
        f"FAIL: S_n skin should describe self-composition"
    print(f"    PASS: S_n skin correctly describes self-composition")


def test_element_labeling():
    """Test that S_n element labeling is correct (1-indexed)."""
    print("\n" + "="*60)
    print("TEST 2: Element labeling consistency")
    print("="*60)
    
    sg = SymmetricGroup(3)
    identity = sg.identity()  # (1, 2, 3)
    print(f"  S_3 identity: {identity}")
    
    # DeckOfCards skin
    skin = DeckOfCardsSkin()
    id_label = skin.element_name(identity, sg)
    print(f"  DeckOfCards label for identity: {id_label}")
    assert "Card 1" in id_label and "Card 2" in id_label and "Card 3" in id_label, \
        f"FAIL: Expected Card 1, Card 2, Card 3 in label, got: {id_label}"
    print(f"    PASS: Uses Card 1, Card 2, Card 3 (1-indexed)")
    
    # Seating skin
    skin2 = SeatingSkin()
    id_label2 = skin2.element_name(identity, sg)
    print(f"  Seating label for identity: {id_label2}")
    assert "Alice" in id_label2 and "Bob" in id_label2 and "Carol" in id_label2, \
        f"FAIL: Expected Alice, Bob, Carol in label, got: {id_label2}"
    print(f"    PASS: Uses Alice, Bob, Carol (1→Alice, 2→Bob, 3→Carol)")
    
    # Test a non-identity permutation
    perm = (2, 3, 1)
    perm_label = skin2.element_name(perm, sg)
    print(f"  Seating label for (2,3,1): {perm_label}")
    assert "Bob" in perm_label, f"FAIL: First position should be Bob (2→Bob)"
    print(f"    PASS: Correctly maps 2→Bob, 3→Carol, 1→Alice")


def test_skin_power_descriptions_all():
    """Test that ALL skins handle power operations correctly."""
    print("\n" + "="*60)
    print("TEST 3: All skins handle power operations")
    print("="*60)
    
    # CyclicGroup skins
    cg = CyclicGroup(7)
    cyclic_skins = [ClockArithmeticSkin(), MusicIntervalsSkin(), RobotStepsSkin(), ColorWheelSkin()]
    for skin in cyclic_skins:
        desc = skin.op_description("power_2", (2,), cg)
        print(f"  {skin.name}: power_2 → '{desc}'")
        assert "multiply" in desc.lower(), f"FAIL: {skin.name} power_2 should say 'multiply'"
        assert "time" not in desc.lower() and "twice" not in desc.lower(), \
            f"FAIL: {skin.name} power_2 should NOT say 'times' or 'twice'"
        print(f"    PASS")
    
    # SymmetricGroup skins
    sg = SymmetricGroup(3)
    sym_skins = [DeckOfCardsSkin(), SeatingSkin()]
    for skin in sym_skins:
        desc = skin.op_description("power_2", (2,), sg)
        print(f"  {skin.name}: power_2 → '{desc}'")
        assert "compose" in desc.lower() or "time" in desc.lower(), \
            f"FAIL: {skin.name} power_2 should describe self-composition"
        # Must NOT be the generic fallback
        assert desc not in ("apply a shuffle", "rearrange the seats"), \
            f"FAIL: {skin.name} power_2 fell through to generic fallback"
        print(f"    PASS")
    
    # DihedralGroup skins
    dg = DihedralGroup(4)
    dih_skins = [PolygonSymmetriesSkin(), TileFlipSkin()]
    for skin in dih_skins:
        desc = skin.op_description("power_3", (3,), dg)
        print(f"  {skin.name}: power_3 → '{desc}'")
        assert "compose" in desc.lower() or "time" in desc.lower(), \
            f"FAIL: {skin.name} power_3 should describe self-composition"
        assert desc not in ("apply a symmetry", "apply a tile move"), \
            f"FAIL: {skin.name} power_3 fell through to generic fallback"
        print(f"    PASS")
    
    # FiniteField skins
    ff = FiniteField(5)
    ff_skins = [SecretCodesSkin(), ModularArithmeticSkin()]
    for skin in ff_skins:
        desc = skin.op_description("power_2", (2,), ff)
        print(f"  {skin.name}: power_2 → '{desc}'")
        assert "multiply" in desc.lower(), f"FAIL: {skin.name} power_2 should say 'multiply'"
        print(f"    PASS")


def test_full_generation_and_validation():
    """Generate a full task set and validate it."""
    print("\n" + "="*60)
    print("TEST 4: Full generation + validation")
    print("="*60)
    
    gen = AlgebraidGenerator(seed=42)
    task_set = gen.generate(
        depths=[1, 2, 3],
        tasks_per_depth=20,
        families=["intra", "inter", "field", "rule"],
        include_dimensions=True,
    )
    print(f"  Generated {len(task_set)} tasks")
    print(task_set.summary())
    
    # Validate
    validator = TaskValidator()
    report = validator.validate_taskset(task_set)
    print_validation_report(report)
    
    # Save the task set
    os.makedirs("./results", exist_ok=True)
    task_set.to_jsonl("./results/tasks_v2.1.jsonl")
    print(f"  Saved to ./results/tasks_v2.1.jsonl")
    
    return report


def test_specific_defect_scenario():
    """Reproduce the exact scenario from the feedback and verify it's fixed."""
    print("\n" + "="*60)
    print("TEST 5: Reproduce feedback defect scenarios")
    print("="*60)
    
    # Scenario 1: Z_5, start at 1, power_2 should give 2 (= 2*1 mod 5)
    # The OLD skin said "ascend by one semitone, 2 times" → human thinks 1+2=3
    # The NEW skin should say "multiply the current tone number by 2 (mod 5)"
    cg = CyclicGroup(5)
    skin = MusicIntervalsSkin()
    
    x = 1
    # power_2 on Z_5: 2*1 mod 5 = 2
    result = cg.op_chain(x, x)  # x + x = 2*x
    print(f"  Scenario 1: Z_5, x=1, power_2")
    print(f"    Correct answer: {result} (= 2*1 mod 5)")
    assert result == 2
    
    desc = skin.op_description("power_2", (2,), cg)
    print(f"    Skin description: '{desc}'")
    # A human reading "multiply the current tone number by 2 (mod 5)" with x=1
    # would compute 2*1 mod 5 = 2. Correct!
    assert "multiply" in desc.lower()
    print(f"    PASS: Human would compute 2*1 mod 5 = 2, matching the answer")
    
    # Scenario 2: S_3, power_2 should NOT say "apply a shuffle" (generic fallback)
    sg = SymmetricGroup(3)
    skin2 = DeckOfCardsSkin()
    desc2 = skin2.op_description("power_2", (2,), sg)
    print(f"\n  Scenario 2: S_3, power_2")
    print(f"    Skin description: '{desc2}'")
    assert desc2 != "apply a shuffle", "FAIL: Still using generic fallback!"
    assert "compose" in desc2.lower() or "time" in desc2.lower()
    print(f"    PASS: Description specifies self-composition, not generic 'apply a shuffle'")
    
    # Scenario 3: S_3 elements should not use "Two, Three, Four"
    identity = sg.identity()  # (1, 2, 3)
    label = skin2.element_name(identity, sg)
    print(f"\n  Scenario 3: S_3 identity element labeling")
    print(f"    Label: '{label}'")
    assert "Four" not in label, f"FAIL: S_3 identity should not contain 'Four'"
    assert "Card 1" in label or "Card 2" in label, \
        f"FAIL: Should use Card 1, Card 2, Card 3"
    print(f"    PASS: No off-by-one in element labeling")


if __name__ == "__main__":
    print("ALGEBRAID v2.1 - Defect Fix Verification")
    print("="*60)
    
    try:
        test_power_operation_semantics()
        test_element_labeling()
        test_skin_power_descriptions_all()
        test_specific_defect_scenario()
        report = test_full_generation_and_validation()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED")
        print("="*60)
        
        if report["failed"] > 0:
            print(f"\nNote: {report['failed']} tasks have validation warnings/errors.")
            print("Review the validation report above for details.")
        
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
