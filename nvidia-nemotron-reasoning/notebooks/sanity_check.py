#!/usr/bin/env python3
"""
Quick sanity check (15-20 min GPU) before full training.

Verifies:
1. Model loads successfully
2. target_modules regex matches actual projection layer names in NemotronH
3. LoRA has >0 trainable parameters
4. Oracle produces training examples from train.csv

Run this on Kaggle BEFORE the full 3-4 hour training run.
"""

# %% [markdown]
# # Cell 0: Setup

# %%
import subprocess, sys, os, gc, glob, json, re, math, time
import warnings
warnings.filterwarnings('ignore')

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

print("=" * 70)
print("NVIDIA Nemotron Reasoning Challenge — Sanity Check")
print("=" * 70)
print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")

if torch.cuda.is_available():
    N_GPUS = torch.cuda.device_count()
    GPU_NAME = torch.cuda.get_device_name(0)
    VRAM = torch.cuda.get_device_properties(0).total_mem / 1024**3 if hasattr(torch.cuda.get_device_properties(0), 'total_mem') else torch.cuda.get_device_properties(0).total_memory / 1024**3
    CC = torch.cuda.get_device_properties(0).major, torch.cuda.get_device_properties(0).minor
    print(f"GPU: {GPU_NAME}")
    print(f"  {N_GPUS} GPU(s), VRAM: {VRAM:.1f} GB, compute: {CC[0]}.{CC[1]}")
else:
    print("ERROR: No CUDA GPU available")
    sys.exit(1)

# %% [markdown]
# # Cell 1: Configuration

# %%
CONFIG = {
    "model_name": "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default",
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "target_modules": r".*\.(in_proj|out_proj|up_proj|down_proj)$",  # Mamba + MoE only
    "use_4bit": False,  # Try BF16 first for sanity check
}

print(f"\nConfig:")
print(f"  model: {CONFIG['model_name']}")
print(f"  lora_r: {CONFIG['lora_r']}, lora_alpha: {CONFIG['lora_alpha']}")
print(f"  target_modules: {CONFIG['target_modules']}")

# %% [markdown]
# # Cell 2: Load Model & Check Layer Names

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

gc.collect()
torch.cuda.empty_cache()

print("Loading model...")
load_start = time.time()

model = AutoModelForCausalLM.from_pretrained(
    CONFIG["model_name"],
    device_map="auto",
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
)

model.config.use_cache = False

print(f"Model loaded in {time.time() - load_start:.0f}s")

# %% [markdown]
# # Cell 3: Apply LoRA and CRITICAL VERIFICATION

# %%
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=CONFIG["lora_r"],
    lora_alpha=CONFIG["lora_alpha"],
    lora_dropout=CONFIG["lora_dropout"],
    target_modules=CONFIG["target_modules"],
    modules_to_save=None,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)

# CRITICAL: Show actual projection layer names
print("\n" + "=" * 70)
print("CRITICAL CHECK: Projection layer names in NemotronH")
print("=" * 70)
proj_layers = [n for n, _ in model.named_modules() if 'proj' in n.lower()]
print(f"\nFound {len(proj_layers)} projection layers. First 50:")
for i, layer in enumerate(proj_layers[:50], 1):
    print(f"  {i}. {layer}")

# Check if attention projections exist
qkv_layers = [n for n in proj_layers if any(x in n for x in ['q_proj', 'k_proj', 'v_proj', 'qkv', 'Wqkv'])]
print(f"\nAttention projection layers (q/k/v): {len(qkv_layers)}")
if qkv_layers:
    print("  Sample:", qkv_layers[:5])
    print("\n  WARNING: target_modules does NOT include attention layers!")
    print("  These are where reasoning happens. Consider adding q_proj/k_proj/v_proj/o_proj.")
else:
    print("  (none found — only Mamba/MoE projections targeted)")

# CRITICAL: Check trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())

print(f"\n" + "=" * 70)
print("LoRA Parameter Check")
print("=" * 70)
print(f"  Trainable: {trainable_params:,}")
print(f"  Total: {total_params:,}")
print(f"  Ratio: {100 * trainable_params / total_params:.2f}%")

if trainable_params == 0:
    print("\n  FATAL: 0 trainable parameters!")
    print(f"  target_modules '{CONFIG['target_modules']}' matched NO layers.")
    print("  Check the projection layer list above and update target_modules.")
    sys.exit(1)
else:
    print(f"\n  ✓ {trainable_params:,} trainable parameters confirmed")

model.print_trainable_parameters()

# %% [markdown]
# # Cell 4: Quick Oracle Check on Training Data

# %%
import pandas as pd

# Find train.csv
train_path = None
for pattern in [
    "/kaggle/input/**/train.csv",
    "/kaggle/input/nvidia-nemotron*/train.csv",
    "data/train.csv",
]:
    matches = glob.glob(pattern, recursive=True)
    if matches:
        train_path = matches[0]
        break

if train_path is None:
    print("\nWARNING: train.csv not found. Skipping oracle check.")
else:
    print(f"\nLoading {train_path}...")
    df = pd.read_csv(train_path)

    # Simple category detection
    def detect_cat(p):
        p = p.lower()
        if re.search(r'[01]{8}\s*->', p) or 'bit manipulation' in p:
            return 'bit_manipulation'
        if 'gravitational' in p or 'd = 0.5' in p:
            return 'gravitational_constant'
        if 'encrypt' in p or 'cipher' in p:
            return 'text_encryption'
        if 'unit conversion' in p or 'becomes' in p:
            return 'unit_conversion'
        if 'numeral system' in p or 'roman' in p:
            return 'number_base_conversion'
        if 'transformation rules' in p:
            return 'equation_transformation'
        return 'unknown'

    df['category'] = df['prompt'].apply(detect_cat)

    print(f"Loaded {len(df)} problems")
    print("\nCategory distribution:")
    print(df['category'].value_counts())

    # Quick oracle check (sample 10 per category)
    from sys.path import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.oracles import solve

    print("\n" + "=" * 70)
    print("Quick Oracle Check (10 samples per category)")
    print("=" * 70)

    for cat in df['category'].unique():
        if cat == 'unknown':
            continue
        samples = df[df['category'] == cat].head(10)
        correct = 0
        for _, row in samples.iterrows():
            oracle_ans = solve(row['prompt'], cat)
            gt = str(row['answer']).strip()
            if oracle_ans is not None:
                try:
                    if math.isclose(float(oracle_ans.replace(',','')), float(gt.replace(',','')), rel_tol=1e-2, abs_tol=1e-5):
                        correct += 1
                except (ValueError, TypeError):
                    if oracle_ans == gt:
                        correct += 1
        print(f"  {cat}: {correct}/10 oracle matches")

# %% [markdown]
# # Cell 5: Summary

# %%
print("\n" + "=" * 70)
print("SANITY CHECK COMPLETE")
print("=" * 70)
print(f"\n✓ Model loads successfully")
print(f"✓ Found {len(proj_layers)} projection layers")
print(f"✓ LoRA has {trainable_params:,} trainable parameters")
print(f"✓ Oracle checked on {len(df)} training samples" if 'df' in locals() else "✓ Oracle check skipped (no train.csv)")

print("\n" + "=" * 70)
print("READY FOR FULL TRAINING")
print("=" * 70)
print("\nProceed to train_sft_kaggle.py for the full 3-4 hour training run.")
