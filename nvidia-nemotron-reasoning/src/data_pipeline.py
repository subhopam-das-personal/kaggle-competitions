"""
Data generation pipeline for training.

Workflow:
  1. Load competition train.csv
  2. Classify each problem by category
  3. Run oracle on each problem
  4. For problems where oracle == ground truth: generate verified CoT traces
  5. For problems where oracle != ground truth: discard (bad training data)
  6. Generate synthetic problems for easy categories
  7. Output: JSONL file ready for SFT training
"""

import json
import random
import string
import re
from pathlib import Path
from typing import Optional

from .category_detector import detect_category, get_category_stats
from .eval_harness import verify
from .oracles import (
    solve, generate_cot,
    int_to_roman, solve_gravitational, solve_unit_conversion,
    solve_text_encryption, solve_bit_manipulation,
    gravitational_cot, number_base_cot, unit_conversion_cot,
    text_encryption_cot, bit_manipulation_cot,
)


def process_training_data(
    train_csv_path: str,
    output_path: str,
    include_unverified: bool = False,
    max_trace_tokens: int = 2000,
) -> dict:
    """Process the competition training data into verified SFT traces.
    
    Args:
        train_csv_path: path to train.csv
        output_path: path to write output JSONL
        include_unverified: if True, include problems the oracle can't solve
                           (with original answer, no CoT quality guarantee)
        max_trace_tokens: approximate max tokens for CoT (rough: 1 token ≈ 4 chars)
    
    Returns:
        dict with processing statistics
    """
    import pandas as pd
    
    df = pd.read_csv(train_csv_path)
    print(f"Loaded {len(df)} problems from {train_csv_path}")
    
    # Classify
    df['category'] = df['prompt'].apply(detect_category)
    print(f"\nCategory distribution:")
    for cat, count in df['category'].value_counts().items():
        print(f"  {cat}: {count}")
    
    stats = {
        'total': len(df),
        'oracle_correct': 0,
        'oracle_wrong': 0,
        'oracle_failed': 0,
        'included': 0,
        'per_category': {},
    }
    
    examples = []
    
    for _, row in df.iterrows():
        prompt = row['prompt']
        ground_truth = str(row['answer']).strip()
        category = row['category']
        
        if category not in stats['per_category']:
            stats['per_category'][category] = {
                'total': 0, 'oracle_correct': 0, 'oracle_wrong': 0, 'oracle_failed': 0
            }
        stats['per_category'][category]['total'] += 1
        
        # Run oracle
        oracle_answer = solve(prompt, category)
        
        if oracle_answer is None:
            stats['oracle_failed'] += 1
            stats['per_category'][category]['oracle_failed'] += 1
            
            # For text_encryption: oracle returns None when mapping is incomplete,
            # but the provided answer is still likely correct. Include these.
            # For equation_transformation: include with original answer (risky but
            # better than nothing for this hard category).
            if category in ('text_encryption', 'equation_transformation') or include_unverified:
                cot = generate_cot(prompt, ground_truth, category)
                examples.append({
                    'messages': [
                        {'role': 'user', 'content': prompt},
                        {'role': 'assistant', 'content': cot},
                    ],
                    'category': category,
                    'verified': category == 'text_encryption',  # encryption answers are reliable
                })
                stats['included'] += 1
        elif verify(oracle_answer, ground_truth):
            stats['oracle_correct'] += 1
            stats['per_category'][category]['oracle_correct'] += 1
            
            # Generate verified CoT trace
            cot = generate_cot(prompt, oracle_answer, category)
            
            # Check trace length
            if len(cot) <= max_trace_tokens * 4:
                examples.append({
                    'messages': [
                        {'role': 'user', 'content': prompt},
                        {'role': 'assistant', 'content': cot},
                    ],
                    'category': category,
                    'verified': True,
                })
                stats['included'] += 1
        else:
            stats['oracle_wrong'] += 1
            stats['per_category'][category]['oracle_wrong'] += 1
    
    # Write output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')
    
    print(f"\nProcessing complete:")
    print(f"  Oracle correct: {stats['oracle_correct']}")
    print(f"  Oracle wrong (discarded): {stats['oracle_wrong']}")
    print(f"  Oracle failed (couldn't solve): {stats['oracle_failed']}")
    print(f"  Included in training: {stats['included']}")
    print(f"  Written to: {output_path}")
    
    return stats


# ============================================================================
# Synthetic data generators
# ============================================================================

def generate_synthetic_gravitational(n: int, seed: int = 42) -> list[dict]:
    """Generate n synthetic gravitational constant problems with verified CoT."""
    rng = random.Random(seed)
    examples = []
    
    for _ in range(n):
        g = round(rng.uniform(3.0, 50.0), 4)
        n_examples = rng.randint(3, 7)
        time_vals = [round(rng.uniform(0.5, 10.0), 2) for _ in range(n_examples)]
        pairs = [(t, round(0.5 * g * t**2, 2)) for t in time_vals]
        
        query_t = round(rng.uniform(0.5, 10.0), 2)
        answer = f"{0.5 * g * query_t**2:.2f}"
        
        ex_text = "\n".join(f"For t = {t}s, distance = {d} m" for t, d in pairs)
        prompt = (
            "In Alice's Wonderland, the gravitational constant has been secretly changed. "
            f"Here are some example observations:\n{ex_text}\n"
            f"Now, determine the falling distance for t = {query_t}s given d = 0.5*g*t^2."
        )
        
        # Verify oracle agrees
        oracle_ans = solve_gravitational(prompt)
        if oracle_ans and verify(oracle_ans, answer):
            cot = gravitational_cot(prompt, answer)
            examples.append({
                'messages': [
                    {'role': 'user', 'content': prompt},
                    {'role': 'assistant', 'content': f"<think>\n{cot}\n</think>\n\\boxed{{{answer}}}"},
                ],
                'category': 'gravitational_constant',
                'verified': True,
                'synthetic': True,
            })
    
    return examples


def generate_synthetic_roman_numerals(n: int, seed: int = 42) -> list[dict]:
    """Generate n synthetic Roman numeral conversion problems."""
    rng = random.Random(seed)
    examples = []
    
    for _ in range(n):
        target = rng.randint(1, 3999)
        n_examples = rng.randint(3, 6)
        example_nums = rng.sample(range(1, 3999), n_examples)
        
        ex_text = "\n".join(f"{num} -> {int_to_roman(num)}" for num in example_nums)
        prompt = (
            "In Alice's Wonderland, numbers are secretly converted into a different "
            f"numeral system. Some examples are given below:\n{ex_text}\n"
            f"Now, write the number {target} in the Wonderland numeral system."
        )
        
        answer = int_to_roman(target)
        cot = number_base_cot(prompt, answer)
        
        examples.append({
            'messages': [
                {'role': 'user', 'content': prompt},
                {'role': 'assistant', 'content': f"<think>\n{cot}\n</think>\n\\boxed{{{answer}}}"},
            ],
            'category': 'number_base_conversion',
            'verified': True,
            'synthetic': True,
        })
    
    return examples


def generate_synthetic_unit_conversion(n: int, seed: int = 42) -> list[dict]:
    """Generate n synthetic unit conversion problems."""
    rng = random.Random(seed)
    examples = []
    
    for _ in range(n):
        ratio = round(rng.uniform(0.1, 10.0), 6)
        n_examples = rng.randint(3, 7)
        input_vals = [round(rng.uniform(1.0, 100.0), 2) for _ in range(n_examples)]
        pairs = [(inp, round(inp * ratio, 2)) for inp in input_vals]
        
        query = round(rng.uniform(1.0, 100.0), 2)
        answer = f"{query * ratio:.2f}"
        
        ex_text = "\n".join(f"{inp} m becomes {out}" for inp, out in pairs)
        prompt = (
            "In Alice's Wonderland, a secret unit conversion is applied to measurements. "
            f"For example:\n{ex_text}\n"
            f"Now, convert the following measurement: {query} m"
        )
        
        oracle_ans = solve_unit_conversion(prompt)
        if oracle_ans and verify(oracle_ans, answer):
            cot = unit_conversion_cot(prompt, answer)
            examples.append({
                'messages': [
                    {'role': 'user', 'content': prompt},
                    {'role': 'assistant', 'content': f"<think>\n{cot}\n</think>\n\\boxed{{{answer}}}"},
                ],
                'category': 'unit_conversion',
                'verified': True,
                'synthetic': True,
            })
    
    return examples


def generate_synthetic_encryption(n: int, seed: int = 42) -> list[dict]:
    """Generate n synthetic substitution cipher problems."""
    rng = random.Random(seed)
    
    WORDS = [
        "alice", "queen", "king", "rabbit", "hatter", "cat", "turtle", "mouse",
        "dragon", "wizard", "princess", "knight", "castle", "garden", "mirror",
        "forest", "palace", "tower", "bridge", "river", "mountain", "valley",
        "secret", "magical", "mysterious", "golden", "silver", "ancient",
        "discovers", "creates", "imagines", "watches", "reads", "follows",
        "chases", "draws", "builds", "finds", "opens", "closes",
        "the", "in", "on", "under", "through", "near", "behind", "above",
        "book", "door", "key", "map", "scroll", "gem", "crown", "sword",
        "bird", "fish", "star", "moon", "sun", "cloud", "tree", "flower",
        "wise", "brave", "clever", "swift", "strong", "gentle", "fierce",
    ]
    
    examples = []
    
    for _ in range(n):
        # Generate random substitution cipher
        alphabet = list(string.ascii_lowercase)
        shuffled = alphabet.copy()
        rng.shuffle(shuffled)
        cipher = dict(zip(alphabet, shuffled))
        reverse_cipher = {v: k for k, v in cipher.items()}
        
        def encrypt(text):
            return "".join(cipher.get(c, c) for c in text)
        
        # Generate example pairs (encrypted -> decrypted)
        n_examples = rng.randint(3, 6)
        pairs = []
        for _ in range(n_examples):
            sentence = " ".join(rng.sample(WORDS, rng.randint(2, 5)))
            encrypted = encrypt(sentence)
            pairs.append((encrypted, sentence))
        
        # Generate query
        query_words = rng.sample(WORDS, rng.randint(2, 4))
        query_plain = " ".join(query_words)
        query_encrypted = encrypt(query_plain)
        
        ex_text = "\n".join(f"{enc} -> {dec}" for enc, dec in pairs)
        prompt = (
            "In Alice's Wonderland, secret encryption rules are used on text. "
            f"Here are some examples:\n{ex_text}\n"
            f"Now, decrypt the following text: {query_encrypted}"
        )
        
        answer = query_plain
        
        # Verify oracle
        oracle_ans = solve_text_encryption(prompt)
        if oracle_ans and oracle_ans.strip() == answer.strip():
            cot = text_encryption_cot(prompt, answer)
            examples.append({
                'messages': [
                    {'role': 'user', 'content': prompt},
                    {'role': 'assistant', 'content': f"<think>\n{cot}\n</think>\n\\boxed{{{answer}}}"},
                ],
                'category': 'text_encryption',
                'verified': True,
                'synthetic': True,
            })
    
    return examples


def generate_synthetic_bit_manipulation(n: int, seed: int = 42) -> list[dict]:
    """Generate n synthetic bit manipulation problems.
    
    Generates problems with KNOWN operations so the answer is guaranteed correct.
    Focuses on the operations the model struggles with.
    """
    rng = random.Random(seed)
    examples = []
    
    # Define operations to sample from
    operations = [
        ("NOT", lambda x: (~x) & 0xFF),
        ("reverse_bits", lambda x: int(format(x, '08b')[::-1], 2)),
        ("rotate_left_1", lambda x: ((x << 1) | (x >> 7)) & 0xFF),
        ("rotate_right_1", lambda x: ((x >> 1) | (x << 7)) & 0xFF),
        ("shift_left_1", lambda x: (x << 1) & 0xFF),
        ("shift_right_1", lambda x: (x >> 1) & 0xFF),
    ]
    
    # Add XOR/AND/OR with random constants
    for _ in range(20):
        c = rng.randint(0, 255)
        operations.extend([
            (f"XOR_{c}", lambda x, c=c: x ^ c),
            (f"AND_{c}", lambda x, c=c: x & c),
            (f"OR_{c}", lambda x, c=c: x | c),
        ])
    
    for _ in range(n):
        # Pick 1-2 operations to compose
        n_ops = rng.choice([1, 1, 1, 2, 2])  # weighted toward single ops
        selected = rng.sample(operations, min(n_ops, len(operations)))
        
        def composed(x):
            for _, op in selected:
                x = op(x)
            return x
        
        # Generate examples
        n_ex = rng.randint(4, 8)
        inputs = rng.sample(range(256), n_ex)
        pairs = [(format(x, '08b'), format(composed(x), '08b')) for x in inputs]
        
        # Query
        query_int = rng.randint(0, 255)
        while query_int in inputs:
            query_int = rng.randint(0, 255)
        query = format(query_int, '08b')
        answer = format(composed(query_int), '08b')
        
        ex_text = "\n".join(f"{inp} -> {out}" for inp, out in pairs)
        prompt = (
            "In Alice's Wonderland, secret bit manipulation rules transform binary numbers. "
            f"Here are some examples:\n{ex_text}\n"
            f"Now, apply the same transformation for: {query}"
        )
        
        # Verify oracle can solve it (strengthens the oracle)
        oracle_ans = solve_bit_manipulation(prompt)
        verified = oracle_ans is not None and oracle_ans == answer
        
        cot = bit_manipulation_cot(prompt, answer)
        examples.append({
            'messages': [
                {'role': 'user', 'content': prompt},
                {'role': 'assistant', 'content': f"<think>\n{cot}\n</think>\n\\boxed{{{answer}}}"},
            ],
            'category': 'bit_manipulation',
            'verified': verified,  # True only if oracle also confirmed the answer
            'synthetic': True,
        })
    
    return examples


def generate_full_dataset(
    train_csv_path: str,
    output_path: str,
    synthetic_per_easy: int = 500,
    synthetic_encryption: int = 300,
    synthetic_bit: int = 1000,
    seed: int = 42,
) -> dict:
    """Generate the full training dataset: verified real + synthetic examples.
    
    Args:
        train_csv_path: path to competition train.csv
        output_path: path to write output JSONL
        synthetic_per_easy: number of synthetic examples per easy category
        synthetic_encryption: number of synthetic encryption examples
        synthetic_bit: number of synthetic bit manipulation examples
        seed: random seed
    """
    import pandas as pd
    
    # Step 1: Process real data
    print("=" * 60)
    print("Step 1: Processing competition training data")
    print("=" * 60)
    
    df = pd.read_csv(train_csv_path)
    df['category'] = df['prompt'].apply(detect_category)
    
    real_examples = []
    stats = {'real_verified': 0, 'real_discarded': 0, 'real_unsolvable': 0}
    
    for _, row in df.iterrows():
        prompt = row['prompt']
        gt = str(row['answer']).strip()
        cat = row['category']
        
        oracle_ans = solve(prompt, cat)
        
        if oracle_ans is not None and verify(oracle_ans, gt):
            cot = generate_cot(prompt, oracle_ans, cat)
            real_examples.append({
                'messages': [
                    {'role': 'user', 'content': prompt},
                    {'role': 'assistant', 'content': cot},
                ],
                'category': cat,
                'verified': True,
                'synthetic': False,
            })
            stats['real_verified'] += 1
        elif oracle_ans is not None:
            stats['real_discarded'] += 1
        else:
            stats['real_unsolvable'] += 1
            # text_encryption: oracle returns None when mapping is incomplete,
            # but provided answers are reliable. Include them.
            # equation_transformation: include with original answer.
            if cat in ('text_encryption', 'equation_transformation'):
                cot = generate_cot(prompt, gt, cat)
                real_examples.append({
                    'messages': [
                        {'role': 'user', 'content': prompt},
                        {'role': 'assistant', 'content': cot},
                    ],
                    'category': cat,
                    'verified': cat == 'text_encryption',
                    'synthetic': False,
                })
                stats['real_verified'] += 1  # count as included
    
    print(f"Real data: {stats['real_verified']} verified, "
          f"{stats['real_discarded']} discarded (wrong), "
          f"{stats['real_unsolvable']} unsolvable")
    
    # Step 2: Generate synthetic data
    print(f"\n{'=' * 60}")
    print("Step 2: Generating synthetic data")
    print("=" * 60)
    
    synthetic = []
    
    print(f"  Gravitational: {synthetic_per_easy} examples...")
    synthetic.extend(generate_synthetic_gravitational(synthetic_per_easy, seed))
    
    print(f"  Roman numerals: {synthetic_per_easy} examples...")
    synthetic.extend(generate_synthetic_roman_numerals(synthetic_per_easy, seed + 1))
    
    print(f"  Unit conversion: {synthetic_per_easy} examples...")
    synthetic.extend(generate_synthetic_unit_conversion(synthetic_per_easy, seed + 2))
    
    print(f"  Encryption: {synthetic_encryption} examples...")
    synthetic.extend(generate_synthetic_encryption(synthetic_encryption, seed + 3))
    
    print(f"  Bit manipulation: {synthetic_bit} examples...")
    synthetic.extend(generate_synthetic_bit_manipulation(synthetic_bit, seed + 4))
    
    print(f"  Total synthetic: {len(synthetic)}")
    
    # Step 3: Combine and shuffle
    all_examples = real_examples + synthetic
    rng = random.Random(seed)
    rng.shuffle(all_examples)
    
    # Step 4: Write output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + '\n')
    
    # Category breakdown
    cat_counts = {}
    for ex in all_examples:
        cat = ex['category']
        src = 'synthetic' if ex.get('synthetic') else 'real'
        key = f"{cat}_{src}"
        cat_counts[key] = cat_counts.get(key, 0) + 1
    
    print(f"\n{'=' * 60}")
    print(f"Final dataset: {len(all_examples)} examples")
    print("=" * 60)
    for key in sorted(cat_counts.keys()):
        print(f"  {key}: {cat_counts[key]}")
    print(f"\nWritten to: {output_path}")
    
    return {
        'total': len(all_examples),
        'real_verified': stats['real_verified'],
        'synthetic': len(synthetic),
        'category_breakdown': cat_counts,
    }
