#!/usr/bin/env python3
"""
Generate submission using oracle-based solvers.

This script:
1. Reads test.csv
2. Uses deterministic oracles to solve each problem
3. Creates submission.csv
4. Submits to Kaggle via API

Expected score: ~0.60-0.70 (oracle baseline)
"""

import sys
import os
import json
import zipfile
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from category_detector import detect_category
from oracles import solve

# Paths
BASE_DIR = Path(__file__).parent.parent
TEST_CSV = BASE_DIR / "data" / "test.csv"
SUBMISSION_CSV = BASE_DIR / "submission.csv"
SUBMISSION_ZIP = BASE_DIR / "submission.zip"

print("=" * 70)
print("NVIDIA Nemotron Reasoning Challenge - Oracle Submission")
print("=" * 70)

# Load test data
df = pd.read_csv(TEST_CSV)
print(f"Loaded {len(df)} test problems")

# Solve each problem
results = []
category_counts = {}
solved = 0
unsolved = 0

for idx, row in df.iterrows():
    prompt = row['prompt']
    test_id = row['id']

    category = detect_category(prompt)
    category_counts[category] = category_counts.get(category, 0) + 1

    answer = solve(prompt, category)

    if answer is not None:
        results.append({'id': test_id, 'answer': answer})
        solved += 1
    else:
        # For unsolvable problems, use a default placeholder
        # These will likely be wrong but we need to submit something
        results.append({'id': test_id, 'answer': '0'})
        unsolved += 1

    if (idx + 1) % 100 == 0:
        print(f"  Processed {idx + 1}/{len(df)}...")

# Create submission DataFrame
submission_df = pd.DataFrame(results)
submission_df.to_csv(SUBMISSION_CSV, index=False)

with zipfile.ZipFile(SUBMISSION_ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write(SUBMISSION_CSV, "submission.csv")

print(f"\nSubmission saved to: {SUBMISSION_CSV}")
print(f"Zipped to: {SUBMISSION_ZIP} ({SUBMISSION_ZIP.stat().st_size / 1024:.1f} KB)")

# Print statistics
print(f"\nCategory distribution in test set:")
for cat, count in sorted(category_counts.items()):
    print(f"  {cat}: {count}")

print(f"\nOracle performance:")
print(f"  Solved: {solved} ({100*solved/len(df):.1f}%)")
print(f"  Unsolvable (placeholder '0'): {unsolved} ({100*unsolved/len(df):.1f}%)")

# Show sample
print(f"\nSample predictions:")
print(submission_df.head(10).to_string())

# Submit to Kaggle
print("\n" + "=" * 70)
print("Submitting to Kaggle...")
print("=" * 70)

# Get Kaggle API token from env
kaggle_token = os.environ.get('KAGGLE_API_TOKEN')
if not kaggle_token:
    # Try to read from .env
    env_file = BASE_DIR / "outputs" / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if line.startswith('KAGGLE_API_TOKEN='):
                    kaggle_token = line.strip().split('=', 1)[1].strip("'\"")
                    break

if not kaggle_token:
    print("ERROR: KAGGLE_API_TOKEN not found!")
    print("Please set KAGGLE_API_TOKEN environment variable")
    sys.exit(1)

# Setup kaggle API — preserve existing username if already configured
kaggle_dir = Path.home() / '.kaggle'
kaggle_dir.mkdir(exist_ok=True)
kaggle_json = kaggle_dir / 'kaggle.json'
if kaggle_json.exists():
    existing = json.loads(kaggle_json.read_text())
    username = existing.get('username', 'subhopamdas')
else:
    username = 'subhopamdas'
kaggle_json.write_text(json.dumps({"username": username, "key": kaggle_token}))
kaggle_json.chmod(0o600)
os.environ['KAGGLE_CONFIG_DIR'] = str(kaggle_dir)

# Submit command (competition requires submission.zip)
import subprocess
result = subprocess.run(
    ['kaggle', 'competitions', 'submit',
     '-c', 'nvidia-nemotron-model-reasoning-challenge',
     '-f', str(SUBMISSION_ZIP),
     '-m', 'Oracle-based baseline submission'],
    capture_output=True,
    text=True
)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

if result.returncode == 0:
    print("\n✓ Submission successful!")
else:
    print(f"\n✗ Submission failed with exit code {result.returncode}")

print("=" * 70)
