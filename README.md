# NVIDIA Nemotron Reasoning Challenge — LoRA Training Pipeline

**Competition:** [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge) (closes 2026-06-15)  
**Model:** `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (hybrid Mamba-2 + Transformer MoE, 30B/3.5B active)  
**Constraint:** LoRA rank ≤ 32, vLLM inference, `temperature=0.0`, `\boxed{}` grading

## Quick Start

### 1. Test oracles locally (no GPU needed)
```bash
python scripts/test_oracles.py

# With competition data:
python scripts/test_oracles.py --train-csv data/train.csv
```

### 2. Generate training data locally
```bash
# Synthetic only (no competition data needed):
python scripts/generate_data.py --synthetic-only --output data/sft_training.jsonl

# With competition data + synthetic:
python scripts/generate_data.py --train-csv data/train.csv --output data/sft_training.jsonl
```

### 3. Train on Kaggle
1. Upload to Kaggle as a notebook: `notebooks/train_sft_kaggle.py`
2. Add inputs:
   - Competition data: `nvidia-nemotron-model-reasoning-challenge`
   - Model: search for `nemotron-3-nano-30b` in Models tab
3. Select GPU: **RTX Pro 6000** (preferred) or T4 x2
4. In Settings → Custom Packages: `pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`
5. Run All → downloads `submission.zip`

### 4. Submit
Upload the `submission.zip` from your notebook output to the competition.

## Project Structure

```
├── src/
│   ├── category_detector.py   # Classifies problems into 6 categories
│   ├── oracles.py             # Deterministic solvers for each category
│   ├── eval_harness.py        # Local eval matching Kaggle metric exactly
│   └── data_pipeline.py       # Full data generation pipeline
├── notebooks/
│   └── train_sft_kaggle.py    # Self-contained Kaggle training notebook
├── scripts/
│   ├── test_oracles.py        # Oracle test suite
│   └── generate_data.py       # CLI data generation
├── data/                      # Generated training data (gitignored)
├── outputs/                   # Research brief & strategy docs
└── requirements.txt
```

## Six Problem Categories

| Category | Oracle Status | Expected Accuracy |
|---|---|---|
| Gravitational Constant | ✅ Fully solved | ~100% |
| Number Base Conversion | ✅ Fully solved | ~100% |
| Unit Conversion | ✅ Fully solved | ~100% |
| Text Encryption | ✅ Solved (some edge cases) | ~96% |
| **Bit Manipulation** | ⚠️ Partial (simple ops only) | ~40-60% |
| **Equation Transformation** | ⚠️ Numeric subset only | ~30-50% |

## Critical Technical Notes

1. **`transformers >= 5.3.0` required** — older versions have a KV cache bug that drops generation from 38 tok/s to 2 tok/s
2. **Do NOT use `trust_remote_code=True`** — the bundled model code has cache bugs
3. **`gradient_checkpointing=False`** — NemotronH doesn't declare support
4. **Exclude MoE router gate from LoRA** — causes training instability
5. **`\boxed{}` regex stops at first `}`** — never put `}` inside the answer

## Recipe Progression

- **Recipe A (this pipeline):** Oracle-verified SFT → expected **0.65-0.72**
- **Recipe B:** + Teacher distillation for hard categories → **0.74-0.79**
- **Recipe C:** + GRPO on hard categories → **0.79-0.83**
- **Recipe D:** + Top-team-style pipeline → **0.83-0.86**

See `outputs/nemotron-reasoning-lora.md` for the full strategy brief.
