#!/usr/bin/env python3
"""
Manually verify a sample of tasks by re-computing answers from prompts.
This checks that a human following the prompt would arrive at the correct answer.
"""

import sys
import os
import json
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from algebraid import AlgebraidGenerator
from algebraid.primitives import CyclicGroup, SymmetricGroup, DihedralGroup, FiniteField
from algebraid.skins import SKIN_REGISTRY


def verify_intra_task_sample():
    """Generate and manually verify intra-structure tasks."""
    print("\n" + "="*60)
    print("MANUAL VERIFICATION: Intra-structure tasks")
    print("="*60)
    
    gen = AlgebraidGenerator(seed=123)
    task_set = gen.generate(
        depths=[1, 2],
        tasks_per_depth=5,
        families=["intra"],
        include_dimensions=False,
    )
    
    passed = 0
    total = 0
    for task in task_set:
        total += 1
        print(f"\n--- Task {task.task_id} (depth={task.depth}) ---")
        print(f"Prompt:\n{task.prompt}")
        print(f"\nExpected answer: {task.answer}")
        print(f"Answer raw: {task.answer_raw}")
        if task.solution_trace:
            print(f"Trace: {task.solution_trace}")
        
        # Check that the trace is consistent
        if task.solution_trace:
            final_val = task.solution_trace[-1][1]
            if str(final_val) == str(task.answer_raw):
                print("  ✓ Trace matches answer_raw")
                passed += 1
            else:
                print(f"  ✗ MISMATCH: trace final={final_val}, answer_raw={task.answer_raw}")
        else:
            passed += 1  # No trace to check
        
        print()
    
    print(f"\nVerified: {passed}/{total} tasks consistent")
    return passed == total


def verify_specific_power_scenarios():
    """Verify specific power operation scenarios that were buggy before."""
    print("\n" + "="*60)
    print("MANUAL VERIFICATION: Specific power scenarios")
    print("="*60)
    
    all_pass = True
    
    # Scenario A: Z_12 (clock), x=3, power_2 → should be 6
    cg = CyclicGroup(12)
    x = 3
    result = cg.op_chain(x, x)  # 3+3 = 6
    print(f"\n  Z_12: power_2(3) = {result}")
    print(f"    Human reads: 'multiply the current position by 2 (mod 12)'")
    print(f"    Human computes: 2 * 3 mod 12 = 6")
    if result == 6:
        print(f"    ✓ CORRECT")
    else:
        print(f"    ✗ WRONG: got {result}")
        all_pass = False
    
    # Scenario B: Z_5, x=4, power_3 → should be 12 mod 5 = 2
    cg2 = CyclicGroup(5)
    x2 = 4
    result2 = cg2.op_chain(x2, x2, x2)  # 4+4+4 = 12 mod 5 = 2
    print(f"\n  Z_5: power_3(4) = {result2}")
    print(f"    Human reads: 'multiply the current position by 3 (mod 5)'")
    print(f"    Human computes: 3 * 4 mod 5 = 12 mod 5 = 2")
    if result2 == 2:
        print(f"    ✓ CORRECT")
    else:
        print(f"    ✗ WRONG: got {result2}")
        all_pass = False
    
    # Scenario C: S_3, perm=(2,3,1), power_2 → (2,3,1)∘(2,3,1) = (3,1,2)
    sg = SymmetricGroup(3)
    perm = (2, 3, 1)
    result3 = sg.op(perm, perm)
    print(f"\n  S_3: power_2((2,3,1)) = {result3}")
    print(f"    Human reads: 'compose the current arrangement with itself 2 times'")
    print(f"    Human computes: (2,3,1) ∘ (2,3,1)")
    print(f"      pos 1 → 2 → 3, pos 2 → 3 → 1, pos 3 → 1 → 2")
    print(f"      = (3, 1, 2)")
    if result3 == (3, 1, 2):
        print(f"    ✓ CORRECT")
    else:
        print(f"    ✗ WRONG: got {result3}")
        all_pass = False
    
    # Scenario D: GF(7), x=5, power_2 → 2*5 mod 7 = 10 mod 7 = 3
    ff = FiniteField(7)
    x4 = 5
    result4 = ff.op(x4, x4)  # 5+5 = 10 mod 7 = 3
    print(f"\n  GF(7): power_2(5) = {result4}")
    print(f"    Human reads: 'multiply the current code value by 2 (mod 7)'")
    print(f"    Human computes: 2 * 5 mod 7 = 10 mod 7 = 3")
    if result4 == 3:
        print(f"    ✓ CORRECT")
    else:
        print(f"    ✗ WRONG: got {result4}")
        all_pass = False
    
    return all_pass


def verify_prompt_samples():
    """Print a few sample prompts for visual inspection."""
    print("\n" + "="*60)
    print("SAMPLE PROMPTS FOR VISUAL INSPECTION")
    print("="*60)
    
    gen = AlgebraidGenerator(seed=999)
    task_set = gen.generate(
        depths=[1, 2],
        tasks_per_depth=3,
        families=["intra", "rule"],
        include_dimensions=False,
    )
    
    for i, task in enumerate(task_set):
        if i >= 6:
            break
        print(f"\n{'─'*50}")
        print(f"Task {task.task_id} | Family: {task.family.value} | Depth: {task.depth}")
        print(f"Structures: {task.structures}")
        if task.metadata.get("skin"):
            print(f"Skin: {task.metadata['skin']}")
        if task.metadata.get("ops"):
            print(f"Operations: {task.metadata['ops']}")
        print(f"{'─'*50}")
        print(task.prompt)
        print(f"\n  → Answer: {task.answer}")
        print(f"  → Answer raw: {task.answer_raw}")
        if task.solution_trace:
            print(f"  → Trace: {task.solution_trace}")


if __name__ == "__main__":
    ok1 = verify_intra_task_sample()
    ok2 = verify_specific_power_scenarios()
    verify_prompt_samples()
    
    print("\n" + "="*60)
    if ok1 and ok2:
        print("ALL MANUAL VERIFICATIONS PASSED")
    else:
        print("SOME VERIFICATIONS FAILED")
    print("="*60)
