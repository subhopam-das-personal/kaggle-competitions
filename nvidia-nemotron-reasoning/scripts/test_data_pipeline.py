#!/usr/bin/env python3
"""
Integration test for the data pipeline JSONL output format.

Runs generate_full_dataset against the real competition train.csv and verifies:
  - Output file exists and is non-empty
  - Every JSONL line is valid JSON with the expected message structure
  - Every assistant message contains \\boxed{...} with no } inside the boxed value

Usage:
    python scripts/test_data_pipeline.py
    python scripts/test_data_pipeline.py --train-csv path/to/train.csv
"""

import sys
import os
import json
import re
import argparse
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_pipeline import generate_full_dataset


def test_pipeline_output_format(train_csv_path: str):
    """Run the full pipeline and verify output structure."""
    print(f"Testing data pipeline against: {train_csv_path}")

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        output_path = tmp.name

    try:
        stats = generate_full_dataset(
            train_csv_path=train_csv_path,
            output_path=output_path,
            synthetic_per_easy=10,   # small counts for speed in testing
            synthetic_encryption=10,
            synthetic_bit=20,
        )

        # File must exist and be non-empty
        assert Path(output_path).exists(), "Output file was not created"
        assert Path(output_path).stat().st_size > 0, "Output file is empty"
        print(f"  ✓ Output file exists: {output_path}")

        # Read and validate every line
        with open(output_path) as f:
            lines = f.readlines()

        assert len(lines) > 0, "Output file has no lines"
        print(f"  ✓ Output has {len(lines)} examples")

        errors = []
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Must be valid JSON
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {i+1}: invalid JSON — {e}")
                continue

            # Must have 'messages' key
            if 'messages' not in record:
                errors.append(f"Line {i+1}: missing 'messages' key")
                continue

            messages = record['messages']
            if len(messages) < 2:
                errors.append(f"Line {i+1}: messages has fewer than 2 entries")
                continue

            # First message must be user
            if messages[0]['role'] != 'user':
                errors.append(f"Line {i+1}: messages[0].role = '{messages[0]['role']}', expected 'user'")

            # Second message must be assistant
            if messages[1]['role'] != 'assistant':
                errors.append(f"Line {i+1}: messages[1].role = '{messages[1]['role']}', expected 'assistant'")

            # Assistant content must contain \boxed{
            content = messages[1].get('content', '')
            if r'\boxed{' not in content:
                errors.append(f"Line {i+1}: assistant content missing \\boxed{{")
                continue

            # The value inside \boxed{...} must not contain }
            boxed_match = re.search(r'\\boxed\{([^}]*)\}', content)
            if boxed_match is None:
                errors.append(f"Line {i+1}: \\boxed{{ found but regex match failed (possibly nested braces)")
            else:
                boxed_value = boxed_match.group(1)
                if '}' in boxed_value:
                    errors.append(f"Line {i+1}: boxed value contains '}}': {repr(boxed_value[:60])}")

        if errors:
            print(f"\n  ✗ {len(errors)} validation errors:")
            for e in errors[:20]:
                print(f"    {e}")
            if len(errors) > 20:
                print(f"    ... and {len(errors) - 20} more")
            return False

        print(f"  ✓ All {len(lines)} records have valid structure")
        print(f"  ✓ All assistant messages contain \\boxed{{...}} with no nested braces")
        return True

    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-csv",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "train.csv"
        ),
        help="Path to competition train.csv"
    )
    args = parser.parse_args()

    if not Path(args.train_csv).exists():
        print(f"ERROR: train.csv not found at {args.train_csv}")
        print("Pass --train-csv <path> to specify the location.")
        return 1

    success = test_pipeline_output_format(args.train_csv)

    print(f"\n{'='*50}")
    if success:
        print("PASSED")
    else:
        print("FAILED")
    print(f"{'='*50}")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
