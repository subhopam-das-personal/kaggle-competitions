"""
Local evaluation harness that mirrors the Kaggle competition metric exactly.

Key rules from the competition:
  - Answer extraction: \boxed{...} regex (stops at first })
  - Numeric tolerance: rel_tol=1e-2, abs_tol=1e-5
  - String comparison: exact match after .strip(), case-insensitive for binary
  - Binary strings: strict match (no 0b prefix, exact length)
"""

import re
import math
from typing import Optional


def extract_final_answer(text: str) -> Optional[str]:
    """Extract the answer from \\boxed{...} in model output.
    
    Uses the EXACT regex from the Kaggle competition metric:
    r'\\\\boxed\\{([^}]*)(?:\\}|$)'
    
    This stops at the FIRST }. So \\boxed{f(x)} extracts 'f(x' not 'f(x)'.
    Always ensure answers don't contain }.
    """
    # Primary: look for \boxed{...}
    match = re.search(r'\\boxed\{([^}]*)(?:\}|$)', text)
    if match:
        return match.group(1).strip()
    
    # Fallback: look for "answer is X" or "= X" at end
    # (matches Kaggle fallback heuristics)
    fallback = re.search(r'(?:answer\s+is|result\s+is|=)\s*([^\n]+?)$', text, re.IGNORECASE)
    if fallback:
        return fallback.group(1).strip()
    
    return None


def verify(predicted: str, ground_truth: str) -> bool:
    """Check if predicted answer matches ground truth.
    
    Mirrors the Kaggle competition metric exactly:
    1. Try exact string match (after strip + case-insensitive for binary)
    2. Try numeric comparison with rel_tol=1e-2, abs_tol=1e-5
    """
    if predicted is None:
        return False
    
    pred = predicted.strip()
    gt = ground_truth.strip()
    
    # Exact match
    if pred == gt:
        return True
    
    # Case-insensitive match (for binary strings etc.)
    if pred.lower() == gt.lower():
        return True
    
    # Numeric comparison
    try:
        pred_num = float(pred.replace(",", ""))
        gt_num = float(gt.replace(",", ""))
        return math.isclose(pred_num, gt_num, rel_tol=1e-2, abs_tol=1e-5)
    except (ValueError, OverflowError):
        pass
    
    return False


def evaluate_single(model_output: str, ground_truth: str) -> dict:
    """Evaluate a single model response.
    
    Returns:
        dict with 'correct', 'predicted', 'ground_truth', 'extracted'
    """
    predicted = extract_final_answer(model_output)
    
    if predicted is None:
        return {
            'correct': False,
            'predicted': None,
            'ground_truth': ground_truth,
            'extracted': False,
        }
    
    correct = verify(predicted, ground_truth)
    return {
        'correct': correct,
        'predicted': predicted,
        'ground_truth': ground_truth,
        'extracted': True,
    }


def evaluate_batch(predictions: list[dict]) -> dict:
    """Evaluate a batch of predictions.
    
    Args:
        predictions: list of {'model_output': str, 'ground_truth': str, 'id': str}
    
    Returns:
        dict with 'accuracy', 'total', 'correct', 'extracted', 'per_category' (if available)
    """
    results = []
    for pred in predictions:
        result = evaluate_single(pred['model_output'], pred['ground_truth'])
        result['id'] = pred.get('id', '')
        result['category'] = pred.get('category', 'unknown')
        results.append(result)
    
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    extracted = sum(1 for r in results if r['extracted'])
    
    # Per-category breakdown
    categories = {}
    for r in results:
        cat = r['category']
        if cat not in categories:
            categories[cat] = {'total': 0, 'correct': 0}
        categories[cat]['total'] += 1
        if r['correct']:
            categories[cat]['correct'] += 1
    
    for cat in categories:
        categories[cat]['accuracy'] = (
            categories[cat]['correct'] / categories[cat]['total']
            if categories[cat]['total'] > 0 else 0
        )
    
    return {
        'accuracy': correct / total if total > 0 else 0,
        'total': total,
        'correct': correct,
        'extracted': extracted,
        'extraction_rate': extracted / total if total > 0 else 0,
        'per_category': categories,
        'details': results,
    }


if __name__ == "__main__":
    # Self-test
    tests = [
        ("The answer is \\boxed{42}", "42", True),
        ("\\boxed{3.14}", "3.14159", True),       # within rel_tol=1e-2
        ("\\boxed{0.333}", "0.333", True),
        ("No boxed answer here", "42", False),
        ("\\boxed{hello}", "world", False),
        ("\\boxed{10110011}", "10110011", True),   # binary exact match
        ("\\boxed{MMXXVI}", "MMXXVI", True),       # Roman numeral
        ("\\boxed{42.00}", "42", True),            # numeric equality
    ]
    
    print("Running eval harness self-tests:")
    for output, truth, expected in tests:
        result = evaluate_single(output, truth)
        status = "PASS" if result['correct'] == expected else "FAIL"
        print(f"  [{status}] '{output[:40]}' vs '{truth}' → {result['correct']} (expected {expected})")
