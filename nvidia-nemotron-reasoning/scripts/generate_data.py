#!/usr/bin/env python3
"""
Generate training data locally (no GPU needed).

Usage:
    python scripts/generate_data.py --train-csv data/train.csv --output data/sft_training.jsonl
    
    # Or just generate synthetic data without competition data:
    python scripts/generate_data.py --synthetic-only --output data/synthetic_training.jsonl
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_pipeline import (
    generate_full_dataset,
    generate_synthetic_gravitational,
    generate_synthetic_roman_numerals,
    generate_synthetic_unit_conversion,
    generate_synthetic_encryption,
    generate_synthetic_bit_manipulation,
)
import json
import random


def main():
    parser = argparse.ArgumentParser(description="Generate SFT training data")
    parser.add_argument("--train-csv", type=str, help="Path to competition train.csv")
    parser.add_argument("--output", type=str, default="data/sft_training.jsonl",
                       help="Output JSONL path")
    parser.add_argument("--synthetic-only", action="store_true",
                       help="Only generate synthetic data (no competition data needed)")
    parser.add_argument("--synth-easy", type=int, default=500,
                       help="Synthetic examples per easy category")
    parser.add_argument("--synth-enc", type=int, default=300,
                       help="Synthetic encryption examples")
    parser.add_argument("--synth-bit", type=int, default=800,
                       help="Synthetic bit manipulation examples")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    if args.synthetic_only:
        print("Generating synthetic-only dataset...")
        all_examples = []
        all_examples.extend(generate_synthetic_gravitational(args.synth_easy, args.seed))
        all_examples.extend(generate_synthetic_roman_numerals(args.synth_easy, args.seed + 1))
        all_examples.extend(generate_synthetic_unit_conversion(args.synth_easy, args.seed + 2))
        all_examples.extend(generate_synthetic_encryption(args.synth_enc, args.seed + 3))
        all_examples.extend(generate_synthetic_bit_manipulation(args.synth_bit, args.seed + 4))
        
        rng = random.Random(args.seed)
        rng.shuffle(all_examples)
        
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, 'w') as f:
            for ex in all_examples:
                f.write(json.dumps(ex) + '\n')
        
        print(f"Written {len(all_examples)} examples to {args.output}")
    else:
        if not args.train_csv:
            print("Error: --train-csv required (or use --synthetic-only)")
            sys.exit(1)
        
        generate_full_dataset(
            train_csv_path=args.train_csv,
            output_path=args.output,
            synthetic_per_easy=args.synth_easy,
            synthetic_encryption=args.synth_enc,
            synthetic_bit=args.synth_bit,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
