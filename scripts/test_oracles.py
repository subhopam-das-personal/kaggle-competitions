#!/usr/bin/env python3
"""
Test the oracle solvers against known inputs.
Run this BEFORE generating any training data to verify correctness.

Usage:
    python scripts/test_oracles.py
    python scripts/test_oracles.py --train-csv data/train.csv  # test against competition data
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.oracles import (
    solve_gravitational, solve_number_base, solve_unit_conversion,
    solve_text_encryption, solve_bit_manipulation, solve, int_to_roman,
)
from src.category_detector import detect_category
from src.eval_harness import verify, extract_final_answer
import math


def test_gravitational():
    """Test gravitational constant oracle."""
    print("Testing gravitational oracle...")
    
    prompt = (
        "In Alice's Wonderland, the gravitational constant has been secretly changed. "
        "Here are some example observations:\n"
        "For t = 1.0s, distance = 4.9 m\n"
        "For t = 2.0s, distance = 19.6 m\n"
        "For t = 3.0s, distance = 44.1 m\n"
        "Now, determine the falling distance for t = 5.0s given d = 0.5*g*t^2."
    )
    # g = 9.8, d = 0.5 * 9.8 * 25 = 122.5
    answer = solve_gravitational(prompt)
    assert answer is not None, "Oracle returned None"
    assert math.isclose(float(answer), 122.5, rel_tol=0.01), f"Expected ~122.5, got {answer}"
    print(f"  ✓ Basic test: {answer}")
    
    return True


def test_roman_numerals():
    """Test Roman numeral oracle."""
    print("Testing Roman numeral oracle...")
    
    test_cases = [
        (1, "I"), (4, "IV"), (9, "IX"), (42, "XLII"),
        (1994, "MCMXCIV"), (3999, "MMMCMXCIX"),
    ]
    
    for num, expected in test_cases:
        result = int_to_roman(num)
        assert result == expected, f"int_to_roman({num}): expected {expected}, got {result}"
        print(f"  ✓ {num} → {result}")
    
    # Test full oracle with prompt
    prompt = (
        "In Alice's Wonderland, numbers are secretly converted into a different "
        "numeral system. Some examples are given below:\n"
        "10 -> X\n"
        "50 -> L\n"
        "100 -> C\n"
        "Now, write the number 42 in the Wonderland numeral system."
    )
    answer = solve_number_base(prompt)
    assert answer == "XLII", f"Expected XLII, got {answer}"
    print(f"  ✓ Full oracle: 42 → {answer}")
    
    return True


def test_unit_conversion():
    """Test unit conversion oracle."""
    print("Testing unit conversion oracle...")
    
    prompt = (
        "In Alice's Wonderland, a secret unit conversion is applied to measurements. "
        "For example:\n"
        "10.0 m becomes 25.0\n"
        "20.0 m becomes 50.0\n"
        "Now, convert the following measurement: 15.0 m"
    )
    # ratio = 2.5, answer = 37.5
    answer = solve_unit_conversion(prompt)
    assert answer is not None, "Oracle returned None"
    assert math.isclose(float(answer), 37.5, rel_tol=0.01), f"Expected ~37.5, got {answer}"
    print(f"  ✓ Basic test: {answer}")
    
    return True


def test_text_encryption():
    """Test text encryption oracle."""
    print("Testing text encryption oracle...")
    
    prompt = (
        "In Alice's Wonderland, secret encryption rules are used on text. "
        "Here are some examples:\n"
        "bcd -> abc\n"
        "efg -> def\n"
        "hij -> ghi\n"
        "Now, decrypt the following text: ceg"
    )
    # Caesar cipher shift by 1: c->b, e->d, g->f
    answer = solve_text_encryption(prompt)
    assert answer is not None, "Oracle returned None"
    assert answer == "bdf", f"Expected 'bdf', got '{answer}'"
    print(f"  ✓ Caesar cipher: {answer}")
    
    return True


def test_bit_manipulation():
    """Test bit manipulation oracle."""
    print("Testing bit manipulation oracle...")
    
    # XOR with 0xFF (NOT)
    prompt = (
        "In Alice's Wonderland, secret bit manipulation rules transform binary numbers. "
        "Here are some examples:\n"
        "00000000 -> 11111111\n"
        "11111111 -> 00000000\n"
        "10101010 -> 01010101\n"
        "Now, apply the same transformation for: 11001100"
    )
    answer = solve_bit_manipulation(prompt)
    assert answer is not None, "Oracle returned None"
    assert answer == "00110011", f"Expected 00110011, got {answer}"
    print(f"  ✓ NOT operation: {answer}")
    
    # XOR with constant
    prompt2 = (
        "Here are some examples:\n"
        "00000000 -> 10101010\n"
        "11111111 -> 01010101\n"
        "11001100 -> 01100110\n"
        "Now, apply the same transformation for: 00001111"
    )
    answer2 = solve_bit_manipulation(prompt2)
    assert answer2 is not None, "Oracle returned None"
    assert answer2 == "10100101", f"Expected 10100101, got {answer2}"
    print(f"  ✓ XOR constant: {answer2}")
    
    return True


def test_category_detector():
    """Test category detection."""
    print("Testing category detector...")
    
    tests = [
        ("gravitational constant has been changed", "gravitational_constant"),
        ("secret encryption rules are used", "text_encryption"),
        ("10.0 m becomes 25.0", "unit_conversion"),
        ("numeral system", "number_base_conversion"),
        ("10110011 -> 01001100", "bit_manipulation"),
        ("transformation rules", "equation_transformation"),
    ]
    
    for prompt, expected_cat in tests:
        cat = detect_category(prompt)
        assert cat == expected_cat, f"Expected '{expected_cat}', got '{cat}' for: {prompt[:40]}"
        print(f"  ✓ '{prompt[:40]}...' → {cat}")
    
    return True


def test_eval_harness():
    """Test the eval harness."""
    print("Testing eval harness...")
    
    tests = [
        ("\\boxed{42}", "42", True),
        ("\\boxed{3.14}", "3.14159", True),
        ("\\boxed{10110011}", "10110011", True),
        ("no boxed", "42", False),
        ("\\boxed{wrong}", "right", False),
    ]
    
    for output, truth, expected in tests:
        extracted = extract_final_answer(output)
        if extracted is None:
            result = False
        else:
            result = verify(extracted, truth)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{output[:30]}' vs '{truth}' → {result}")
        assert result == expected, f"Expected {expected}, got {result}"
    
    return True


def test_against_competition_data(train_csv_path):
    """Test oracles against actual competition data."""
    import pandas as pd
    
    print(f"\nTesting against competition data: {train_csv_path}")
    df = pd.read_csv(train_csv_path)
    
    # Classify
    df['category'] = df['prompt'].apply(detect_category)
    print(f"Total problems: {len(df)}")
    print(f"Category distribution:\n{df['category'].value_counts().to_string()}")
    
    # Test oracle per category
    results = {}
    for _, row in df.iterrows():
        cat = row['category']
        if cat not in results:
            results[cat] = {'total': 0, 'correct': 0, 'wrong': 0, 'failed': 0}
        results[cat]['total'] += 1
        
        oracle_ans = solve(row['prompt'], cat)
        gt = str(row['answer']).strip()
        
        if oracle_ans is None:
            results[cat]['failed'] += 1
        elif verify(oracle_ans, gt):
            results[cat]['correct'] += 1
        else:
            results[cat]['wrong'] += 1
    
    print(f"\nOracle accuracy per category:")
    print(f"{'Category':<30} {'Total':>6} {'Correct':>8} {'Wrong':>6} {'Failed':>7} {'Acc':>7}")
    print("-" * 70)
    for cat in sorted(results.keys()):
        r = results[cat]
        acc = r['correct'] / r['total'] * 100 if r['total'] > 0 else 0
        print(f"{cat:<30} {r['total']:>6} {r['correct']:>8} {r['wrong']:>6} {r['failed']:>7} {acc:>6.1f}%")
    
    total = sum(r['total'] for r in results.values())
    correct = sum(r['correct'] for r in results.values())
    print(f"{'OVERALL':<30} {total:>6} {correct:>8} {'':>6} {'':>7} {correct/total*100:>6.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=str, help="Test against competition data")
    args = parser.parse_args()
    
    passed = 0
    failed = 0
    
    for test_fn in [
        test_category_detector,
        test_gravitational,
        test_roman_numerals,
        test_unit_conversion,
        test_text_encryption,
        test_bit_manipulation,
        test_eval_harness,
    ]:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*50}")
    
    if args.train_csv:
        test_against_competition_data(args.train_csv)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
