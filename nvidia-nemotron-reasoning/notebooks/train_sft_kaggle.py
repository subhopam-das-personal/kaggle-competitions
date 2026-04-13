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

# Strategy: prefer transformers >= 5.3.0 (native NemotronH, no mamba-ssm needed)
# This is the correct path for new GPUs like Blackwell (sm_120) where mamba-ssm
# has no prebuilt wheels and silent PyPI installs produce incompatible binaries.
USE_TRUST_REMOTE_CODE = False  # Set True only if native transformers path is unavailable

import importlib.metadata as _meta
_tf_ver_str = _meta.version("transformers")
_tf_ver = tuple(int(x) for x in _tf_ver_str.split('.')[:2])

if _tf_ver >= (5, 3):
    print(f"✓ transformers {_tf_ver_str} — native NemotronH (no mamba-ssm needed)")
else:
    print(f"transformers {_tf_ver_str} < 5.3.0, upgrading for native NemotronH support...")
    _result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "transformers>=5.3.0"],
        capture_output=True, text=True, timeout=300
    )
    # Check new version via subprocess to avoid stale import cache
    _ver_check = subprocess.run(
        [sys.executable, "-c", "import transformers; print(transformers.__version__)"],
        capture_output=True, text=True
    )
    _new_ver = _ver_check.stdout.strip()
    _new_tf_ver = tuple(int(x) for x in _new_ver.split('.')[:2]) if _new_ver else (0, 0)
    if _new_tf_ver >= (5, 3):
        print(f"✓ transformers upgraded to {_new_ver} — native NemotronH, no mamba-ssm needed")
    else:
        print(f"⚠ transformers upgrade failed (still {_new_ver or _tf_ver_str}), falling back to mamba-ssm path")
        if _result.stderr:
            print(f"  Upgrade error: {_result.stderr[-400:]}")
        USE_TRUST_REMOTE_CODE = True

# Install mamba-ssm ONLY if the native transformers path is unavailable
if USE_TRUST_REMOTE_CODE:
    print("Installing mamba-ssm and causal-conv1d (required for transformers < 5.3.0 path)...")

    _OFFLINE_DIRS = [
        "/kaggle/input/datasets/dennisfong/nvidia-nemotron-offline-packages/offline_packages/",
        "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script/",
    ]
    _mamba_installed = False
    for _odir in _OFFLINE_DIRS:
        if os.path.exists(_odir):
            _wheels = glob.glob(f"{_odir}/**/*causal_conv1d*.whl", recursive=True)
            _mamba_wheels = glob.glob(f"{_odir}/**/*mamba_ssm*.whl", recursive=True)
            if _wheels and _mamba_wheels:
                _res = subprocess.run([sys.executable, "-m", "pip", "install", "--no-index",
                    f"--find-links={_odir}", "causal-conv1d", "mamba-ssm"],
                    capture_output=True, text=True)
                if _res.returncode == 0:
                    _mamba_installed = True
                    print(f"✓ Installed from offline packages at {_odir}")
                break

    if not _mamba_installed:
        print("Offline packages not found, installing from PyPI...")
        _res = subprocess.run([sys.executable, "-m", "pip", "install",
            "causal-conv1d", "mamba-ssm"], capture_output=True, text=True, timeout=600)
        if _res.returncode == 0:
            _mamba_installed = True
            print("✓ Installed from PyPI")
        else:
            print(f"✗ PyPI install failed: {_res.stderr[-300:]}")

    # Verify mamba-ssm actually imports (PyPI may install incompatible wheel silently)
    try:
        import mamba_ssm
        import causal_conv1d
        print("✓ mamba_ssm imported successfully")
    except ImportError as _e:
        # Last resort: build from source for the specific GPU compute capability
        _cc = torch.cuda.get_device_properties(0)
        _arch = f"{_cc.major}.{_cc.minor}"
        print(f"Wheels incompatible ({_e}), building from source for sm_{_cc.major}{_cc.minor}...")
        _env = os.environ.copy()
        _env["TORCH_CUDA_ARCH_LIST"] = _arch
        _res = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-build-isolation",
             "git+https://github.com/Dao-AILab/causal-conv1d.git",
             "git+https://github.com/state-spaces/mamba.git"],
            capture_output=True, text=True, timeout=1200, env=_env
        )
        try:
            import mamba_ssm
            import causal_conv1d
            print(f"✓ mamba_ssm compiled from source for sm_{_cc.major}{_cc.minor}")
        except ImportError:
            raise RuntimeError(
                f"Cannot load NemotronH: transformers>=5.3.0 upgrade failed AND "
                f"mamba-ssm failed to build for sm_{_cc.major}{_cc.minor}.\n"
                f"Compile errors:\n{_res.stderr[-500:]}\n"
                f"Fix: ensure internet access, or add 'nvidia-nemotron-offline-packages' as input."
            )

pip_install([
    "peft>=0.14.0",
    "trl>=0.17.0",
    "datasets",
    "accelerate",
    "einops",
    "bitsandbytes",
    "kagglehub",
])

import transformers
print(f"transformers: {transformers.__version__}")
print(f"USE_TRUST_REMOTE_CODE: {USE_TRUST_REMOTE_CODE}")
print("✓ All dependencies ready")

# %% [markdown]
# # Cell 2: Configuration

# %%
CONFIG = {
    # Model
    "model_name": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    
    # LoRA — MUST be rank ≤ 32 per competition rules
    # Evaluator uses max_lora_rank=32 and max_tokens=7680
    "lora_r": 32,
    "lora_alpha": 32,
    "lora_dropout": 0.0,  # 0 at eval → train with 0 for consistency
    # Proven target modules for NemotronH (Mamba + MoE layers only, avoids router)
    # Do NOT use "all-linear" — evaluator's vLLM rejects adapters targeting router/lm_head
    "target_modules": r".*\.(in_proj|out_proj|up_proj|down_proj)$",
    
    # Training
    # Match evaluator max_tokens=7680 to avoid truncating CoT at inference time
    "max_seq_length": 7680,
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

# Load model via kagglehub (same as the competition evaluator)
# This downloads/caches the official competition model
try:
    import kagglehub
    MODEL_PATH = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")
    print(f"✓ Model path (kagglehub): {MODEL_PATH}")
except Exception as _e:
    print(f"⚠ kagglehub failed ({_e}), falling back to local scan")
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

def solve_equation_transformation_local(prompt):
    """Solve equation transformation via symbol bijection search."""
    import math as _math
    from itertools import permutations as _perms

    parts = re.split(
        r'(?:Now,?\s*)?(?:determine|find|compute|calculate)\s+the\s+result\s+(?:for|of)[:\s]*',
        prompt, flags=re.IGNORECASE
    )
    if len(parts) < 2:
        return None

    examples_text, query_text = parts[0], parts[1].strip()
    equations = re.findall(r'([^\n=]+?)\s*=\s*([^\n]+)', examples_text)
    equations = [
        (lhs.strip(), rhs.strip()) for lhs, rhs in equations
        if not any(w in lhs.lower() for w in ['wonderland', 'secret', 'below', 'rule', 'example'])
    ]
    if not equations:
        return None

    all_text = " ".join(lhs + " " + rhs for lhs, rhs in equations) + " " + query_text
    symbols = sorted(set(c for c in all_text
                         if c not in ' +-*/=(){}[].,\n\t' and not c.isdigit()))
    if not symbols:
        return None

    # Detect operator-position symbols
    op_symbols = set()
    for lhs, _ in equations:
        tokens = lhs.split()
        if len(tokens) == 3 and tokens[1] in symbols:
            op_symbols.add(tokens[1])

    digit_syms = [s for s in symbols if s not in op_symbols]
    op_syms = [s for s in symbols if s in op_symbols]

    def apply_map(expr, dm, om={}):
        out = ""
        for c in expr:
            if c in dm: out += str(dm[c])
            elif c in om: out += om[c]
            elif c in ' +-*/()' or c.isdigit(): out += c
            else: return None
        return out

    def safe_eval(expr):
        try:
            cleaned = expr.strip()
            if not re.match(r'^[\d\s+\-*/().]+$', cleaned):
                return None
            return float(eval(cleaned, {"__builtins__": {}}, {}))
        except Exception:
            return None

    def fmt(v):
        return str(int(v)) if v == int(v) else f"{v:.4f}".rstrip('0').rstrip('.')

    operators = ['+', '-', '*', '/']
    digits = list(range(10))
    max_count = 100000
    count = 0

    op_perms = list(_perms(operators, min(len(op_syms), len(operators)))) if op_syms else [()]
    for op_perm in op_perms:
        om = dict(zip(op_syms, op_perm)) if op_syms else {}
        for digit_perm in _perms(digits, len(digit_syms)):
            count += 1
            if count > max_count:
                return None
            dm = dict(zip(digit_syms, digit_perm))
            valid = True
            for lhs, rhs in equations:
                try:
                    ld = apply_map(lhs, dm, om)
                    rd = apply_map(rhs, dm, om)
                    if ld is None or rd is None: valid = False; break
                    lv = safe_eval(ld); rv = safe_eval(rd)
                    if lv is None or rv is None: valid = False; break
                    if not _math.isclose(lv, rv, rel_tol=1e-6): valid = False; break
                except Exception:
                    valid = False; break
            if valid:
                r = apply_map(query_text, dm, om)
                if r is not None:
                    rv = safe_eval(r)
                    if rv is not None:
                        return fmt(rv)
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
        'equation_transformation': solve_equation_transformation_local,
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
    """Generate step-by-step CoT for equation transformation.

    Explicitly teaches: extract symbol table → substitute → evaluate.
    """
    parts = re.split(
        r'(?:Now,?\s*)?(?:determine|find|compute|calculate)\s+the\s+result\s+(?:for|of)[:\s]*',
        prompt, flags=re.IGNORECASE
    )
    examples_text = parts[0] if parts else prompt
    query_text = parts[1].strip() if len(parts) > 1 else ""

    equations = re.findall(r'([^\n=]+?)\s*=\s*([^\n]+)', examples_text)
    equations = [
        (lhs.strip(), rhs.strip()) for lhs, rhs in equations
        if not any(w in lhs.lower() for w in ['wonderland', 'secret', 'below', 'rule', 'example'])
    ]

    all_text = " ".join(lhs + " " + rhs for lhs, rhs in equations) + " " + query_text
    symbols = sorted(set(c for c in all_text
                         if c not in ' +-*/=(){}[].,\n\t' and not c.isdigit()))

    lines = [
        "I need to decode the symbol mapping from the given examples.",
        "",
        f"Step 1: Unique symbols found: {', '.join(symbols) if symbols else '(none)'}",
        "",
        "Step 2: Determine the bijection by testing assignments against each example:",
    ]
    for lhs, rhs in equations[:4]:
        lines.append(f"  {lhs} = {rhs}")

    lines.extend([
        "",
        "Step 3: Apply the discovered mapping to the query expression.",
        f"  Query: {query_text}",
        f"  After substitution and evaluation: {answer}",
        "",
        "Step 4: The result is:",
        f"  {answer}",
    ])
    return "\n".join(lines)

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

def gen_synthetic_eq_transform(n, rng):
    """Generate equation_transformation examples with proper CoT.

    Each example uses a random digit bijection (and optionally operator bijection).
    Produces problems that look like the competition format.
    """
    import unicodedata
    # Greek letters as symbol pool (same as competition)
    GREEK = list("αβγδεζηθικλμνξοπρστυφχψω")
    OPERATORS = ['+', '-', '*']  # avoid '/' to keep integer answers
    out = []
    for _ in range(n):
        n_symbols = rng.randint(2, 5)
        chosen = rng.sample(GREEK, n_symbols)

        # Randomly decide if one symbol maps to an operator
        use_op_sym = rng.random() < 0.3 and n_symbols >= 3
        if use_op_sym:
            op_sym = rng.choice(chosen[1:])  # don't make the first sym an operator
            digit_syms = [s for s in chosen if s != op_sym]
            op_val = rng.choice(OPERATORS)
        else:
            op_sym = None
            digit_syms = chosen
            op_val = None

        # Assign digit values (1-9, avoid 0 to prevent divide issues)
        available_digits = list(range(1, 10))
        rng.shuffle(available_digits)
        digit_map = dict(zip(digit_syms, available_digits[:len(digit_syms)]))

        def encode(val):
            # Find which symbol maps to this digit
            for s, v in digit_map.items():
                if v == val:
                    return s
            return str(val)

        def apply(expr_parts):
            """expr_parts is list of (sym_or_digit, is_operator)"""
            result = ""
            for tok in expr_parts:
                if tok in digit_map:
                    result += str(digit_map[tok])
                elif tok == op_sym:
                    result += op_val if op_val else tok
                elif tok in '+-*':
                    result += tok
                else:
                    result += str(tok)
            return result

        # Generate example equations: A op B = C
        n_examples = rng.randint(3, 5)
        examples = []
        for _ in range(n_examples):
            a_sym = rng.choice(digit_syms)
            b_sym = rng.choice(digit_syms)
            a_val = digit_map[a_sym]
            b_val = digit_map[b_sym]

            if use_op_sym:
                op_char = op_val
                lhs_str = f"{a_sym} {op_sym} {b_sym}"
            else:
                op_char = rng.choice(OPERATORS)
                lhs_str = f"{a_sym} {op_char} {b_sym}"

            try:
                rhs_val = int(eval(f"{a_val} {op_char} {b_val}"))
            except Exception:
                continue
            examples.append((lhs_str, str(rhs_val)))

        if not examples:
            continue

        # Query
        q_a = rng.choice(digit_syms)
        q_b = rng.choice(digit_syms)
        qa_val = digit_map[q_a]
        qb_val = digit_map[q_b]
        if use_op_sym:
            q_op_char = op_val
            query_str = f"{q_a} {op_sym} {q_b}"
        else:
            q_op_char = rng.choice(OPERATORS)
            query_str = f"{q_a} {q_op_char} {q_b}"

        try:
            answer_val = int(eval(f"{qa_val} {q_op_char} {qb_val}"))
        except Exception:
            continue
        answer = str(answer_val)

        ex_lines = "\n".join(f"{lhs} = {rhs}" for lhs, rhs in examples)
        prompt = (
            "In Alice's Wonderland, transformation rules map symbols to digits and operators. "
            "The examples below follow a consistent bijection:\n"
            f"{ex_lines}\n"
            f"Now, determine the result for: {query_str}"
        )
        cot = generate_cot(prompt, answer, 'equation_transformation')
        out.append({
            'messages': [
                {'role': 'user', 'content': prompt},
                {'role': 'assistant', 'content': cot},
            ],
            'category': 'equation_transformation',
        })
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
        # Oracle disagrees with ground truth
        if cat == 'equation_transformation':
            # For equation_transformation, accept ground truth (oracle can't solve this category)
            cot = generate_cot(prompt, gt, cat)
            all_data.append({
                'messages': [
                    {'role': 'user', 'content': prompt},
                    {'role': 'assistant', 'content': cot},
                ],
                'category': cat,
            })
            unsolvable_count += 1
        else:
            discarded_count += 1  # Oracle disagrees with ground truth for other categories
    else:
        unsolvable_count += 1
        # text_encryption: oracle returns None when mapping is incomplete,
        # but provided answers are still correct. Include them.
        # equation_transformation: include with original answer (oracle returns None)
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
# SYNTH_EQ disabled: competition's equation_transformation is arbitrary custom operations
# (e.g., 34/44=1 where / is per-problem custom rule), not digit bijections.
# Synthetic bijection data would teach the wrong pattern.

synth = []
synth.extend(gen_synthetic_grav(SYNTH_EASY, rng))
synth.extend(gen_synthetic_roman(SYNTH_EASY, rng))
synth.extend(gen_synthetic_unit(SYNTH_EASY, rng))
synth.extend(gen_synthetic_enc(SYNTH_ENC, rng))
synth.extend(gen_synthetic_bit(SYNTH_BIT, rng))
# synth.extend(gen_synthetic_eq_transform(SYNTH_EQ, rng))  # Disabled - wrong problem type

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

# USE_TRUST_REMOTE_CODE is set by the dependency cell above.
# False (preferred): transformers >= 5.3.0 native NemotronH — fast KV cache
# True (fallback): bundled modeling_nemotron_h.py — requires mamba-ssm

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

gc.collect()
torch.cuda.empty_cache()

print("Loading model...")
if USE_TRUST_REMOTE_CODE:
    print("  (trust_remote_code=True — using bundled modeling_nemotron_h.py, transformers native unavailable)")
    print("  ⚠ Note: bundled impl may have slower KV cache than native transformers>=5.3.0")
else:
    print("  (trust_remote_code=False — using native NemotronH from transformers)")
load_start = time.time()

if USE_4BIT:
    from transformers import BitsAndBytesConfig
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        trust_remote_code=USE_TRUST_REMOTE_CODE,
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
        trust_remote_code=USE_TRUST_REMOTE_CODE,
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
    # Regex targets in_proj/out_proj (Mamba) + up_proj/down_proj (MoE experts)
    # This is the proven-working pattern for NemotronH hybrid architecture
    target_modules=CONFIG["target_modules"],
    modules_to_save=None,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)

# Debug: show actual attention layer names (run this BEFORE assuming target_modules is right)
proj_layers = [n for n, _ in model.named_modules() if 'proj' in n.lower()]
print(f"Projection layers (first 40): {proj_layers[:40]}")

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

# FATAL guard: if 0 trainable params, target_modules matched nothing
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
assert trainable_params > 0, (
    f"FATAL: LoRA adapter has 0 trainable parameters. "
    f"target_modules regex '{CONFIG['target_modules']}' matched no layers. "
    f"Check proj_layers above and update target_modules to match."
)
print(f"✓ {trainable_params:,} trainable parameters confirmed")

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
    packing=True,
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
print(f"DONE! submission.zip saved at: {submission_path}")
print(f"{'='*70}")

# %% [markdown]
# # Cell 10: Auto-Submit to Kaggle (uses KAGGLE_KEY secret)

# %%
# Reads KAGGLE_KEY from Kaggle notebook secrets (Settings → Add secret → KAGGLE_KEY).
# Set the secret value to your Kaggle API key (from https://www.kaggle.com/settings).
# If the secret is absent, prints instructions and skips — training output is still saved.

print("\n" + "=" * 70)
print("Auto-submitting to Kaggle...")
print("=" * 70)

try:
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    kaggle_key = secrets.get_secret("KAGGLE_KEY")
    kaggle_user = secrets.get_secret("KAGGLE_USER") if "KAGGLE_USER" in dir(secrets) else "subhopamdas"
except Exception:
    kaggle_key = os.environ.get("KAGGLE_KEY", "")
    kaggle_user = os.environ.get("KAGGLE_USER", "subhopamdas")

if not kaggle_key:
    print("⚠ KAGGLE_KEY secret not set.")
    print("  To enable auto-submit: Settings → Secrets → Add secret → KAGGLE_KEY = <your api key>")
    print(f"  Submission zip is at: {submission_path}")
    print("  Download it and run:")
    print(f"    kaggle competitions submit -c nvidia-nemotron-model-reasoning-challenge -f submission.zip -m 'SFT LoRA rank-32'")
else:
    import json as _json
    from pathlib import Path as _Path
    _kaggle_dir = _Path.home() / ".kaggle"
    _kaggle_dir.mkdir(exist_ok=True)
    _kaggle_json = _kaggle_dir / "kaggle.json"
    # Preserve existing username if credentials already configured
    if _kaggle_json.exists():
        _existing = _json.loads(_kaggle_json.read_text())
        _user = _existing.get("username", kaggle_user)
    else:
        _user = kaggle_user
    _kaggle_json.write_text(_json.dumps({"username": _user, "key": kaggle_key}))
    _kaggle_json.chmod(0o600)

    _result = subprocess.run(
        ["kaggle", "competitions", "submit",
         "-c", "nvidia-nemotron-model-reasoning-challenge",
         "-f", str(submission_path),
         "-m", f"SFT LoRA rank-{CONFIG['lora_r']} — oracle-verified CoT"],
        capture_output=True, text=True
    )
    print(_result.stdout)
    if _result.stderr:
        print("STDERR:", _result.stderr)
    if _result.returncode == 0:
        print("✓ Submission successful!")
    else:
        print(f"✗ Auto-submit failed (exit {_result.returncode}). Download submission.zip manually.")
print("=" * 70)
