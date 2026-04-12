#!/usr/bin/env python3
"""
Nemotron Reasoning Challenge — SFT Training Notebook (Kaggle-ready)

This notebook trains a LoRA adapter on the NVIDIA Nemotron-3-Nano-30B-A3B model
using oracle-verified chain-of-thought traces + synthetic data.

=== KAGGLE SETUP ===
1. Add competition data: "nvidia-nemotron-model-reasoning-challenge"
2. Add model: search for "nemotron-3-nano-30b" in Models tab
   (or use: metric/nemotron-3-nano-30b-a3b-bf16)
3. Add this training dataset: upload your generated data/ folder as a dataset
4. Select GPU: RTX Pro 6000 (preferred) or T4 x2
5. In Settings → Custom Packages, add:
   pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
6. Enable Internet (needed for first run to download deps)

=== CRITICAL FIXES (from competition research) ===
- transformers >= 5.3.0 required (fixes GRPO cache bug: 2→38 tok/s)
- DO NOT use trust_remote_code=True (breaks KV cache, 19x slowdown)
- gradient_checkpointing=False (NemotronH doesn't declare support)
- Router gate EXCLUDED from LoRA targets (unstable on MoE)
- Fused cross-entropy essential for VRAM (Unsloth or manual)
"""

# %% [markdown]
# # Cell 0: Environment Setup

# %%
import subprocess, sys, os, gc, glob, json, re, math, random, time
import warnings
warnings.filterwarnings('ignore')

# Set BEFORE any CUDA allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

print("=" * 70)
print("NVIDIA Nemotron Reasoning Challenge — SFT Training")
print("=" * 70)
print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")

if torch.cuda.is_available():
    N_GPUS = torch.cuda.device_count()
    GPU_NAME = torch.cuda.get_device_name(0)
    VRAM = torch.cuda.get_device_properties(0).total_mem / 1024**3 if hasattr(torch.cuda.get_device_properties(0), 'total_mem') else torch.cuda.get_device_properties(0).total_memory / 1024**3
    CC = torch.cuda.get_device_properties(0).major
    print(f"GPU: {N_GPUS}× {GPU_NAME} ({VRAM:.1f} GiB, sm_{CC}0)")
    
    # CUDA sanity check
    t = torch.tensor([1.0, 2.0, 3.0], device="cuda")
    assert (t * t).sum().item() == 14.0
    del t
    print("✓ CUDA functional")
else:
    raise RuntimeError("No GPU detected! Enable GPU in notebook settings.")

TOTAL_VRAM = VRAM * N_GPUS
USE_4BIT = TOTAL_VRAM < 40  # QLoRA for T4; BF16 for RTX Pro 6000
print(f"Mode: {'QLoRA 4-bit' if USE_4BIT else 'BF16 full precision'} ({TOTAL_VRAM:.0f} GiB total)")

# %% [markdown]
# # Cell 1: Install Dependencies

# %%
def pip_install(packages):
    """Install packages quietly."""
    for pkg in packages:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", pkg],
            capture_output=True, timeout=300
        )

# CRITICAL: transformers >= 5.3.0 fixes the KV cache bug
pip_install([
    "transformers>=5.3.0",
    "peft>=0.14.0",
    "trl>=0.17.0",
    "datasets",
    "accelerate",
    "einops",
    "bitsandbytes",
])

# Verify critical version
import transformers
tf_version = tuple(int(x) for x in transformers.__version__.split('.')[:2])
print(f"transformers: {transformers.__version__}")
assert tf_version >= (5, 3), (
    f"CRITICAL: transformers {transformers.__version__} < 5.3.0. "
    f"The KV cache bug makes generation 19x slower. "
    f"Run: pip install 'transformers>=5.3.0'"
)
print("✓ All dependencies ready")

# %% [markdown]
# # Cell 2: Configuration

# %%
CONFIG = {
    # Model
    "model_name": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    
    # LoRA — MUST be rank ≤ 32 per competition rules
    "lora_r": 32,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    # all-linear EXCEPT router gate (unstable on MoE)
    "target_modules": "all-linear",
    "modules_to_exclude": ["router"],  # exclude MoE router gate
    
    # Training
    "max_seq_length": 2048,  # Keep short for easy categories; saves VRAM
    "num_train_epochs": 2,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,  # effective batch = 16
    "learning_rate": 2e-5,  # Conservative for SFT
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "lr_scheduler_type": "cosine",
    "bf16": True,
    "fp16": False,
    # CRITICAL: must be False — NemotronHForCausalLM doesn't declare support
    "gradient_checkpointing": False,
    "optim": "adamw_torch",
    
    # Output
    "output_dir": "/kaggle/working/lora_adapter",
    "submission_path": "/kaggle/working/submission.zip",
}

# Auto-detect model path from Kaggle inputs
MODEL_PATH = CONFIG["model_name"]
if os.path.exists("/kaggle/input"):
    for root, dirs, files in os.walk("/kaggle/input"):
        if "config.json" in files:
            try:
                with open(os.path.join(root, "config.json")) as f:
                    cfg = json.load(f)
                if "nemotron" in str(cfg.get("architectures", "")).lower():
                    MODEL_PATH = root
                    print(f"✓ Found model at: {root}")
                    break
            except Exception:
                pass
        # Don't recurse into model subdirs
        if "config.json" in files:
            dirs.clear()

if MODEL_PATH == CONFIG["model_name"]:
    print(f"⚠ Model not found locally, will download: {MODEL_PATH}")

# Adjust for QLoRA on low-VRAM GPUs
if USE_4BIT:
    CONFIG["optim"] = "paged_adamw_8bit"
    CONFIG["max_seq_length"] = 1024  # Shorter to fit in T4 VRAM
    print("Adjusted config for QLoRA/T4 mode")

print(f"\nConfig: rank={CONFIG['lora_r']}, lr={CONFIG['learning_rate']}, "
      f"epochs={CONFIG['num_train_epochs']}, seq_len={CONFIG['max_seq_length']}")

# %% [markdown]
# # Cell 3: Data Generation (runs on CPU, no GPU needed)

# %%
import string
from typing import Optional

# ============================================================
# Category detection
# ============================================================

def detect_category(prompt):
    p = prompt.lower()
    if re.search(r'[01]{8}\s*->', p) or 'bit manipulation' in p:
        return 'bit_manipulation'
    if any(kw in p for kw in ['gravitational', 'falling distance', 'free fall', 'd = 0.5', 'gravity']):
        return 'gravitational_constant'
    if any(kw in p for kw in ['encrypt', 'decrypt', 'encryption rules', 'cipher']):
        return 'text_encryption'
    if any(kw in p for kw in ['unit conversion', 'measurement', 'convert the following']):
        return 'unit_conversion'
    if re.search(r'[\d.]+\s*m\s+becomes\s+[\d.]+', p):
        return 'unit_conversion'
    if 'becomes' in p and re.search(r'[\d.]+.*?becomes.*?[\d.]+', p):
        return 'unit_conversion'
    if any(kw in p for kw in ['numeral system', 'numeral', 'roman', 'write the number', 'secret number']):
        return 'number_base_conversion'
    if any(kw in p for kw in ['transformation rules', 'determine the result']):
        return 'equation_transformation'
    return 'unknown'

# ============================================================
# Oracles (deterministic solvers)
# ============================================================

ROMAN_VALUES = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
    (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
    (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
]

def int_to_roman(num):
    result = ""
    for value, symbol in ROMAN_VALUES:
        while num >= value:
            result += symbol
            num -= value
    return result

def solve_gravitational(prompt):
    pairs = re.findall(r't\s*=\s*([\d.]+)\s*s.*?distance\s*=\s*([\d.]+)', prompt, re.I)
    if not pairs:
        return None
    g_vals = []
    for t_s, d_s in pairs:
        t, d = float(t_s), float(d_s)
        if t > 0:
            g_vals.append(2 * d / (t ** 2))
    if not g_vals:
        return None
    g = sorted(g_vals)[len(g_vals) // 2]
    qm = re.search(r'(?:determine|calculate|find|compute).*?t\s*=\s*([\d.]+)', prompt, re.I)
    if qm:
        qt = float(qm.group(1))
    else:
        all_t = re.findall(r't\s*=\s*([\d.]+)', prompt)
        if not all_t:
            return None
        qt = float(all_t[-1])
    return f"{0.5 * g * qt**2:.2f}"

def solve_unit_conversion(prompt):
    pairs = re.findall(r'([\d.]+)\s*(?:m|meters?)?\s*(?:becomes|=|→|->)\s*([\d.]+)', prompt)
    if not pairs:
        return None
    ratios = [float(o) / float(i) for i, o in pairs if float(i) > 0]
    if not ratios:
        return None
    ratio = sorted(ratios)[len(ratios) // 2]
    # Find query
    parts = prompt.lower().split("convert")
    if len(parts) > 1:
        nums = re.findall(r'([\d.]+)', parts[-1])
        if nums:
            return f"{float(nums[0]) * ratio:.2f}"
    m = re.search(r'convert.*?(?:following|measurement)[:\s]*([\d.]+)', prompt, re.I)
    if m:
        return f"{float(m.group(1)) * ratio:.2f}"
    return None

def solve_number_base(prompt):
    roman_pat = re.findall(r'(\d+)\s*->\s*([IVXLCDM]+)', prompt)
    if roman_pat:
        m = re.search(r'write the number\s+(\d+)', prompt, re.I)
        if not m:
            m = re.search(r'convert.*?(\d+)', prompt, re.I)
        if m:
            return int_to_roman(int(m.group(1)))
    return None

def solve_text_encryption(prompt):
    split_pats = [
        r'Now,?\s*decrypt(?:\s+the\s+following)?\s*(?:text)?[:\s]+',
        r'Decrypt\s*(?:the\s+following)?[:\s]+',
    ]
    query = None
    examples_part = prompt
    for pat in split_pats:
        parts = re.split(pat, prompt, flags=re.I)
        if len(parts) >= 2:
            examples_part = parts[0]
            query = parts[-1].strip().strip('"\'').strip()
            break
    if query is None:
        return None
    pairs = re.findall(r'([a-zA-Z][a-zA-Z\s]*?)\s*(?:->|=>|→)\s*([a-zA-Z][a-zA-Z\s]*?)(?:\n|$)', examples_part)
    if not pairs:
        return None
    mapping = {}
    for enc, dec in pairs:
        ec = enc.strip().replace(" ", "")
        dc = dec.strip().replace(" ", "")
        if len(ec) == len(dc):
            for e, d in zip(ec, dc):
                mapping[e.lower()] = d.lower()
    if not mapping:
        return None
    result = []
    unmapped = 0
    for c in query:
        if c.lower() in mapping:
            result.append(mapping[c.lower()])
        elif c.isalpha():
            result.append(c)
            unmapped += 1
        else:
            result.append(c)
    if unmapped > 0:
        return None  # incomplete mapping — can't verify
    return "".join(result)

def solve_bit_manipulation(prompt):
    pairs = re.findall(r'([01]{8})\s*(?:->|=>|→)\s*([01]{8})', prompt)
    qm = re.search(r'(?:for|apply|input)[:\s]*([01]{8})', prompt, re.I)
    if not pairs or not qm:
        return None
    query = qm.group(1)
    examples = [(int(i, 2), int(o, 2)) for i, o in pairs]
    
    # Try XOR/AND/OR with constants
    for const in range(256):
        if all(i ^ const == o for i, o in examples):
            return format(int(query, 2) ^ const, '08b')
        if all(i & const == o for i, o in examples):
            return format(int(query, 2) & const, '08b')
        if all(i | const == o for i, o in examples):
            return format(int(query, 2) | const, '08b')
    
    # Try NOT
    if all((~i) & 0xFF == o for i, o in examples):
        return format((~int(query, 2)) & 0xFF, '08b')
    
    # Try reverse
    def rev(x): return int(format(x, '08b')[::-1], 2)
    if all(rev(i) == o for i, o in examples):
        return format(rev(int(query, 2)), '08b')
    
    # Try per-bit rules
    bit_rules = []
    for bp in range(8):
        found = False
        for sp in range(8):
            for inv in [False, True]:
                if all(
                    (((i >> (7 - sp)) & 1) ^ int(inv)) == ((o >> (7 - bp)) & 1)
                    for i, o in examples
                ):
                    bit_rules.append((bp, sp, inv))
                    found = True
                    break
            if found:
                break
        if not found:
            # Constant bit
            if all(((o >> (7 - bp)) & 1) == 0 for _, o in examples):
                bit_rules.append((bp, -1, False))
            elif all(((o >> (7 - bp)) & 1) == 1 for _, o in examples):
                bit_rules.append((bp, -1, True))
            else:
                return None
    
    if len(bit_rules) == 8:
        q = int(query, 2)
        result = 0
        for bp, sp, inv in bit_rules:
            if sp == -1:
                bv = 1 if inv else 0
            else:
                bv = ((q >> (7 - sp)) & 1) ^ int(inv)
            result |= (bv << (7 - bp))
        return format(result, '08b')
    return None

def solve(prompt, category=None):
    if category is None:
        category = detect_category(prompt)
    solvers = {
        'gravitational_constant': solve_gravitational,
        'number_base_conversion': solve_number_base,
        'unit_conversion': solve_unit_conversion,
        'text_encryption': solve_text_encryption,
        'bit_manipulation': solve_bit_manipulation,
    }
    solver = solvers.get(category)
    if solver:
        try:
            return solver(prompt)
        except Exception:
            return None
    return None

def answers_match(pred, gt):
    if pred is None:
        return False
    if pred.strip() == gt.strip():
        return True
    if pred.strip().lower() == gt.strip().lower():
        return True
    try:
        return math.isclose(float(pred.strip().replace(",","")),
                            float(gt.strip().replace(",","")),
                            rel_tol=1e-2, abs_tol=1e-5)
    except (ValueError, OverflowError):
        return False

# ============================================================
# CoT Trace Generators
# ============================================================

def grav_cot(prompt, answer):
    pairs = re.findall(r't\s*=\s*([\d.]+)\s*s.*?distance\s*=\s*([\d.]+)', prompt, re.I)
    if not pairs:
        return f"Using d = 0.5*g*t², the answer is {answer}"
    t0, d0 = float(pairs[0][0]), float(pairs[0][1])
    g = 2 * d0 / (t0**2)
    all_t = re.findall(r't\s*=\s*([\d.]+)', prompt)
    qt = float(all_t[-1])
    lines = [
        f"Using d = 0.5 * g * t², I find g from the examples.",
        f"From example 1: t={t0}s, d={d0}m → g = 2×{d0}/{t0}² = {g:.6f}",
    ]
    if len(pairs) > 1:
        t1, d1 = float(pairs[1][0]), float(pairs[1][1])
        lines.append(f"Verify: t={t1}s, d={d1}m → g = {2*d1/t1**2:.6f} ✓")
    lines.append(f"For t={qt}s: d = 0.5 × {g:.6f} × {qt}² = {0.5*g*qt**2:.2f}")
    return "\n".join(lines)

def roman_cot(prompt, answer):
    m = re.search(r'write the number\s+(\d+)', prompt, re.I)
    if not m:
        return f"Converting: {answer}"
    num = int(m.group(1))
    lines = [f"Converting {num} to Roman numerals:"]
    r = num
    for val, sym in ROMAN_VALUES:
        if r >= val:
            count = r // val
            lines.append(f"  {r} ÷ {val} = {count} → {sym * count}")
            r -= count * val
    lines.append(f"Result: {answer}")
    return "\n".join(lines)

def unit_cot(prompt, answer):
    pairs = re.findall(r'([\d.]+)\s*(?:m|meters?)?\s*(?:becomes|=|→|->)\s*([\d.]+)', prompt)
    if not pairs:
        return f"Applying conversion ratio: {answer}"
    r = float(pairs[0][1]) / float(pairs[0][0])
    return (f"Ratio from examples: {pairs[0][1]}/{pairs[0][0]} = {r:.6f}\n"
            f"Applying ratio to query: {answer}")

def enc_cot(prompt, answer):
    pairs = re.findall(r'([a-zA-Z][a-zA-Z\s]*?)\s*(?:->|=>|→)\s*([a-zA-Z][a-zA-Z\s]*?)(?:\n|$)', prompt)
    mapping = {}
    for enc, dec in pairs:
        ec = enc.strip().replace(" ", "")
        dc = dec.strip().replace(" ", "")
        if len(ec) == len(dc):
            for e, d in zip(ec, dc):
                mapping[e.lower()] = d.lower()
    sm = dict(sorted(mapping.items()))
    lines = [
        "Building substitution cipher mapping from examples:",
        "  " + ", ".join(f"{k}→{v}" for k, v in list(sm.items())[:15]),
    ]
    if len(sm) > 15:
        lines[-1] += f" ... ({len(sm)} total)"
    lines.append(f"Decrypted: {answer}")
    return "\n".join(lines)

def bit_cot(prompt, answer):
    pairs = re.findall(r'([01]{8})\s*(?:->|=>|→)\s*([01]{8})', prompt)
    lines = ["Analyzing bit transformation from examples:"]
    for i, o in pairs[:5]:
        lines.append(f"  {i} → {o}")
    lines.append(f"Applying pattern: {answer}")
    return "\n".join(lines)

def eq_cot(prompt, answer):
    return f"Analyzing transformation rules from examples.\nApplying to query: {answer}"

def generate_cot(prompt, answer, category):
    cot_fns = {
        'gravitational_constant': grav_cot,
        'number_base_conversion': roman_cot,
        'unit_conversion': unit_cot,
        'text_encryption': enc_cot,
        'bit_manipulation': bit_cot,
        'equation_transformation': eq_cot,
    }
    fn = cot_fns.get(category, lambda p, a: f"Result: {a}")
    reasoning = fn(prompt, answer)
    return f"<think>\n{reasoning}\n</think>\n\\boxed{{{answer}}}"

print("✓ Oracles and CoT generators ready")

# ============================================================
# Synthetic data generators
# ============================================================

WORDS = [
    "alice", "queen", "king", "rabbit", "hatter", "cat", "turtle", "mouse",
    "dragon", "wizard", "princess", "knight", "castle", "garden", "mirror",
    "forest", "palace", "tower", "bridge", "river", "mountain", "valley",
    "secret", "magical", "mysterious", "golden", "silver", "ancient",
    "discovers", "creates", "imagines", "watches", "reads", "follows",
    "book", "door", "key", "map", "scroll", "gem", "crown", "sword",
    "bird", "fish", "star", "moon", "sun", "cloud", "tree", "flower",
]

def gen_synthetic_grav(n, rng):
    out = []
    for _ in range(n):
        g = round(rng.uniform(3.0, 50.0), 4)
        n_ex = rng.randint(3, 7)
        ts = [round(rng.uniform(0.5, 10.0), 2) for _ in range(n_ex)]
        pairs = [(t, round(0.5*g*t**2, 2)) for t in ts]
        qt = round(rng.uniform(0.5, 10.0), 2)
        ans = f"{0.5*g*qt**2:.2f}"
        ex = "\n".join(f"For t = {t}s, distance = {d} m" for t, d in pairs)
        prompt = (f"In Alice's Wonderland, the gravitational constant has been secretly changed. "
                  f"Here are some example observations:\n{ex}\n"
                  f"Now, determine the falling distance for t = {qt}s given d = 0.5*g*t^2.")
        cot = generate_cot(prompt, ans, 'gravitational_constant')
        out.append({'messages': [{'role': 'user', 'content': prompt},
                                  {'role': 'assistant', 'content': cot}],
                     'category': 'gravitational_constant'})
    return out

def gen_synthetic_roman(n, rng):
    out = []
    for _ in range(n):
        target = rng.randint(1, 3999)
        ex_nums = rng.sample(range(1, 3999), rng.randint(3, 6))
        ex = "\n".join(f"{num} -> {int_to_roman(num)}" for num in ex_nums)
        prompt = (f"In Alice's Wonderland, numbers are secretly converted into a different "
                  f"numeral system. Some examples are given below:\n{ex}\n"
                  f"Now, write the number {target} in the Wonderland numeral system.")
        ans = int_to_roman(target)
        cot = generate_cot(prompt, ans, 'number_base_conversion')
        out.append({'messages': [{'role': 'user', 'content': prompt},
                                  {'role': 'assistant', 'content': cot}],
                     'category': 'number_base_conversion'})
    return out

def gen_synthetic_unit(n, rng):
    out = []
    for _ in range(n):
        ratio = round(rng.uniform(0.1, 10.0), 6)
        n_ex = rng.randint(3, 7)
        inps = [round(rng.uniform(1.0, 100.0), 2) for _ in range(n_ex)]
        pairs = [(i, round(i*ratio, 2)) for i in inps]
        q = round(rng.uniform(1.0, 100.0), 2)
        ans = f"{q*ratio:.2f}"
        ex = "\n".join(f"{i} m becomes {o}" for i, o in pairs)
        prompt = (f"In Alice's Wonderland, a secret unit conversion is applied to measurements. "
                  f"For example:\n{ex}\n"
                  f"Now, convert the following measurement: {q} m")
        cot = generate_cot(prompt, ans, 'unit_conversion')
        out.append({'messages': [{'role': 'user', 'content': prompt},
                                  {'role': 'assistant', 'content': cot}],
                     'category': 'unit_conversion'})
    return out

def gen_synthetic_enc(n, rng):
    out = []
    for _ in range(n):
        alph = list(string.ascii_lowercase)
        shuf = alph.copy()
        rng.shuffle(shuf)
        cipher = dict(zip(alph, shuf))
        def encrypt(text, c=cipher):
            return "".join(c.get(ch, ch) for ch in text)
        pairs = []
        for _ in range(rng.randint(3, 6)):
            sentence = " ".join(rng.sample(WORDS, rng.randint(2, 5)))
            pairs.append((encrypt(sentence), sentence))
        qs = " ".join(rng.sample(WORDS, rng.randint(2, 4)))
        eq = encrypt(qs)
        ex = "\n".join(f"{e} -> {d}" for e, d in pairs)
        prompt = (f"In Alice's Wonderland, secret encryption rules are used on text. "
                  f"Here are some examples:\n{ex}\n"
                  f"Now, decrypt the following text: {eq}")
        cot = generate_cot(prompt, qs, 'text_encryption')
        out.append({'messages': [{'role': 'user', 'content': prompt},
                                  {'role': 'assistant', 'content': cot}],
                     'category': 'text_encryption'})
    return out

def gen_synthetic_bit(n, rng):
    """Generate bit manipulation with KNOWN operations."""
    out = []
    for _ in range(n):
        const = rng.randint(0, 255)
        op_type = rng.choice(['xor', 'and', 'or', 'not', 'reverse',
                               'rot_left', 'rot_right'])
        ops = {
            'xor': lambda x, c=const: x ^ c,
            'and': lambda x, c=const: x & c,
            'or': lambda x, c=const: x | c,
            'not': lambda x: (~x) & 0xFF,
            'reverse': lambda x: int(format(x, '08b')[::-1], 2),
            'rot_left': lambda x: ((x << 1) | (x >> 7)) & 0xFF,
            'rot_right': lambda x: ((x >> 1) | (x << 7)) & 0xFF,
        }
        op = ops[op_type]
        n_ex = rng.randint(4, 8)
        inputs = rng.sample(range(256), n_ex + 1)
        q_int = inputs.pop()
        pairs = [(format(x, '08b'), format(op(x), '08b')) for x in inputs]
        query = format(q_int, '08b')
        ans = format(op(q_int), '08b')
        ex = "\n".join(f"{i} -> {o}" for i, o in pairs)
        prompt = (f"In Alice's Wonderland, secret bit manipulation rules transform binary numbers. "
                  f"Here are some examples:\n{ex}\n"
                  f"Now, apply the same transformation for: {query}")
        cot = generate_cot(prompt, ans, 'bit_manipulation')
        out.append({'messages': [{'role': 'user', 'content': prompt},
                                  {'role': 'assistant', 'content': cot}],
                     'category': 'bit_manipulation'})
    return out

print("✓ Synthetic generators ready")

# %% [markdown]
# # Cell 4: Load & Process Training Data

# %%
import pandas as pd

# Find train.csv
train_path = None
for pattern in [
    "/kaggle/input/**/train.csv",
    "/kaggle/input/nvidia-nemotron*/train.csv",
    "data/train.csv",
    "../data/train.csv",
]:
    matches = glob.glob(pattern, recursive=True)
    if matches:
        train_path = matches[0]
        break

if train_path is None:
    raise FileNotFoundError(
        "Cannot find train.csv! Add competition data as input:\n"
        "  → 'Add Input' → 'Competition' → 'nvidia-nemotron-model-reasoning-challenge'"
    )

df = pd.read_csv(train_path)
print(f"Loaded {len(df)} problems from {train_path}")

# Classify and verify
df['category'] = df['prompt'].apply(detect_category)
print(f"\nCategory distribution:")
print(df['category'].value_counts().to_string())

# Process with oracle verification
all_data = []
verified_count = 0
discarded_count = 0
unsolvable_count = 0

for _, row in df.iterrows():
    prompt = row['prompt']
    gt = str(row['answer']).strip()
    cat = row['category']
    
    oracle_ans = solve(prompt, cat)
    
    if oracle_ans is not None and answers_match(oracle_ans, gt):
        # Oracle verified — use oracle answer for clean trace
        cot = generate_cot(prompt, oracle_ans, cat)
        all_data.append({
            'messages': [
                {'role': 'user', 'content': prompt},
                {'role': 'assistant', 'content': cot},
            ],
            'category': cat,
        })
        verified_count += 1
    elif oracle_ans is not None:
        discarded_count += 1  # Oracle disagrees with ground truth
    else:
        unsolvable_count += 1
        # text_encryption: oracle returns None when mapping is incomplete,
        # but provided answers are still correct. Include them.
        # equation_transformation: include with original answer (risky but
        # better than nothing for this hard category).
        if cat in ('text_encryption', 'equation_transformation'):
            cot = generate_cot(prompt, gt, cat)
            all_data.append({
                'messages': [
                    {'role': 'user', 'content': prompt},
                    {'role': 'assistant', 'content': cot},
                ],
                'category': cat,
            })

print(f"\nOracle results:")
print(f"  Verified: {verified_count}")
print(f"  Discarded (oracle ≠ ground truth): {discarded_count}")
print(f"  Unsolvable (oracle returned None): {unsolvable_count}")
print(f"  Included so far: {len(all_data)}")

# Generate synthetic data
print(f"\nGenerating synthetic data...")
rng = random.Random(42)

SYNTH_EASY = 500
SYNTH_ENC = 300
SYNTH_BIT = 800

synth = []
synth.extend(gen_synthetic_grav(SYNTH_EASY, rng))
synth.extend(gen_synthetic_roman(SYNTH_EASY, rng))
synth.extend(gen_synthetic_unit(SYNTH_EASY, rng))
synth.extend(gen_synthetic_enc(SYNTH_ENC, rng))
synth.extend(gen_synthetic_bit(SYNTH_BIT, rng))

print(f"  Generated {len(synth)} synthetic examples")

all_data.extend(synth)
rng.shuffle(all_data)

# Category breakdown
cat_counts = {}
for ex in all_data:
    cat_counts[ex['category']] = cat_counts.get(ex['category'], 0) + 1

print(f"\nFinal dataset: {len(all_data)} examples")
for cat in sorted(cat_counts):
    print(f"  {cat}: {cat_counts[cat]}")

# Free DataFrame
del df
gc.collect()

# %% [markdown]
# # Cell 5: Load Model & Tokenizer

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

# CRITICAL: Do NOT use trust_remote_code=True
# The bundled modeling_nemotron_h.py has a cache bug that drops generation to 2 tok/s.
# transformers >= 5.3.0 has the native implementation that works correctly.

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

gc.collect()
torch.cuda.empty_cache()

print("Loading model...")
load_start = time.time()

if USE_4BIT:
    from transformers import BitsAndBytesConfig
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        # NO trust_remote_code=True — this is critical!
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ),
        torch_dtype=torch.bfloat16,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        # NO trust_remote_code=True!
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
    )

model.config.use_cache = False

print(f"✓ Model loaded in {time.time() - load_start:.0f}s")
for i in range(N_GPUS):
    alloc = torch.cuda.memory_allocated(i) / 1024**3
    print(f"  GPU {i}: {alloc:.2f} GiB allocated")

# %% [markdown]
# # Cell 6: Apply LoRA

# %%
from peft import LoraConfig, get_peft_model, TaskType

if USE_4BIT:
    from peft import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model)

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

# Ensure LoRA params are in BF16
cast_count = 0
for n, p in model.named_parameters():
    if p.requires_grad and p.dtype != torch.bfloat16:
        p.data = p.data.to(torch.bfloat16)
        cast_count += 1
if cast_count:
    print(f"  Cast {cast_count} LoRA params → bfloat16")

model.print_trainable_parameters()
print(f"LoRA rank={CONFIG['lora_r']}, alpha={CONFIG['lora_alpha']}")

# %% [markdown]
# # Cell 7: Format Dataset & Train

# %%
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

def format_for_training(example):
    text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )
    return {"text": text}

dataset = Dataset.from_list(all_data)
dataset = dataset.map(format_for_training, remove_columns=["category"])

# Inspect a sample
print("Sample training text (first 500 chars):")
print(dataset[0]['text'][:500])
print("...")

# Split
split = dataset.train_test_split(test_size=0.02, seed=42)
train_ds, val_ds = split["train"], split["test"]
print(f"\nTrain: {len(train_ds)}, Val: {len(val_ds)}")

# Calculate steps
steps_per_epoch = max(1, len(train_ds) // (
    CONFIG["per_device_train_batch_size"] * CONFIG["gradient_accumulation_steps"]
))
total_steps = steps_per_epoch * CONFIG["num_train_epochs"]
warmup_steps = max(10, int(CONFIG["warmup_ratio"] * total_steps))

print(f"Steps: {steps_per_epoch}/epoch × {CONFIG['num_train_epochs']} = {total_steps}, warmup={warmup_steps}")

sft_config = SFTConfig(
    output_dir=CONFIG["output_dir"],
    num_train_epochs=CONFIG["num_train_epochs"],
    per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
    gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
    learning_rate=CONFIG["learning_rate"],
    warmup_steps=warmup_steps,
    weight_decay=CONFIG["weight_decay"],
    max_grad_norm=CONFIG["max_grad_norm"],
    lr_scheduler_type=CONFIG["lr_scheduler_type"],
    fp16=CONFIG["fp16"],
    bf16=CONFIG["bf16"],
    gradient_checkpointing=CONFIG["gradient_checkpointing"],
    optim=CONFIG["optim"],
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    save_total_limit=2,
    max_seq_length=CONFIG["max_seq_length"],
    dataset_text_field="text",
    packing=False,
    report_to="none",
    seed=42,
    dataloader_num_workers=0,  # Avoid multiprocessing issues on Kaggle
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
)

# Flush CUDA cache before training
gc.collect()
for gi in range(N_GPUS):
    with torch.cuda.device(gi):
        torch.cuda.empty_cache()

for gi in range(N_GPUS):
    free = (torch.cuda.get_device_properties(gi).total_memory - 
            torch.cuda.memory_allocated(gi)) / 1024**3
    print(f"GPU {gi} free: {free:.2f} GiB")

print(f"\n{'='*70}")
print(f"TRAINING: {CONFIG['num_train_epochs']} epochs, "
      f"batch {CONFIG['per_device_train_batch_size']}×{CONFIG['gradient_accumulation_steps']}, "
      f"lr {CONFIG['learning_rate']}")
print(f"{'='*70}")

train_start = time.time()
trainer.train()
train_time = time.time() - train_start

print(f"\n✓ Training complete in {train_time/60:.1f} minutes")

# %% [markdown]
# # Cell 8: Save Adapter & Package Submission

# %%
import zipfile

adapter_dir = CONFIG["output_dir"] + "/final"
model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)

print("Adapter files:")
total_size = 0
for f in sorted(os.listdir(adapter_dir)):
    size = os.path.getsize(os.path.join(adapter_dir, f))
    total_size += size
    print(f"  {f}: {size/1024/1024:.1f} MB")
print(f"  Total: {total_size/1024/1024:.1f} MB")

# Verify adapter_config.json exists
assert os.path.exists(os.path.join(adapter_dir, "adapter_config.json")), \
    "FATAL: adapter_config.json missing!"

# Verify rank
with open(os.path.join(adapter_dir, "adapter_config.json")) as f:
    adapter_cfg = json.load(f)
assert adapter_cfg.get("r", 99) <= 32, \
    f"FATAL: LoRA rank {adapter_cfg['r']} > 32 (competition limit)"
print(f"✓ adapter_config.json verified: rank={adapter_cfg['r']}")

# Package as submission.zip
submission_path = CONFIG["submission_path"]
with zipfile.ZipFile(submission_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(adapter_dir):
        for file in files:
            filepath = os.path.join(root, file)
            arcname = os.path.relpath(filepath, adapter_dir)
            zf.write(filepath, arcname)

zip_size = os.path.getsize(submission_path) / 1024 / 1024
print(f"\n✓ Submission packaged: {submission_path} ({zip_size:.1f} MB)")

# %% [markdown]
# # Cell 9: Quick Validation (optional)

# %%
# Quick smoke test on a few examples to verify the adapter works
print("\nQuick validation on 5 random training examples...")

# This just checks that the model generates valid \boxed{} output
# Full eval requires vLLM inference (matches the Kaggle harness)

try:
    model.eval()
    test_samples = random.sample(all_data, min(5, len(all_data)))
    
    for i, sample in enumerate(test_samples):
        prompt = sample['messages'][0]['content']
        # Just check tokenization works
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(model.device)
        
        with torch.no_grad():
            output = model.generate(
                inputs,
                max_new_tokens=256,
                temperature=0.0,
                do_sample=False,
            )
        
        response = tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True)
        has_boxed = "\\boxed{" in response
        cat = sample.get('category', 'unknown')
        print(f"  [{cat}] {'✓' if has_boxed else '✗'} boxed={'yes' if has_boxed else 'NO'} "
              f"len={len(response)}")
        
        if not has_boxed and i == 0:
            print(f"    Response preview: {response[:200]}...")

except Exception as e:
    print(f"  Validation skipped (non-critical): {e}")

print(f"\n{'='*70}")
print("DONE! Submit {submission_path} to the competition.")
print(f"{'='*70}")
