# Research File R1 — Base Model, Tooling, Kaggle Harness

**Status:** T1 done (single-session, inline execution).

## 1. What "Nemotron-3-Nano-30B" actually is

| Fact | Value | Source |
|---|---|---|
| Canonical ID | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (also `-FP8`, `-Base-BF16`) | [S1] |
| Architecture | **Hybrid Mamba-2 + Transformer MoE**. 52 layers: 23 Mamba-2 + 23 MoE + 6 GQA attention (GQA with 2 groups). | [S1] |
| Params | **31.8 B total / 3.5–3.6 B active** per forward pass ("A3B" = Active-3B) | [S1] |
| Experts | **128 routed + 1 shared** per MoE layer, **top-6 routed** per token | [S1] |
| Hidden dim (H) | 2,688 | [S11] |
| Vocab | 131,072 | [S11] |
| Expert FFN intermediate | 1,856 routed, 3,712 shared | [S11] |
| Pretraining | **25 T tokens**, WSD schedule, peak LR 1e-3, min 1e-5, 8 B-token warmup, batch 3072 | [S1] |
| Context length | 1 M tokens (default HF config 256 k; set `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` for 1 M) | [S1] |
| Released | **Dec 15, 2025** | [S1] |
| License | **NVIDIA Nemotron Open Model License** (permissive for commercial + derivative use, with safety provisions) | [S1] |
| Technical report | arXiv 2512.20848, Dec 23 2025 | [S2] |

### Thinking mode
- Unified reasoning + non-reasoning model: a single checkpoint with `enable_thinking=True/False` flag in the chat template (and a `/think` slash in the system prompt). [S1]
- Reasoning trace is wrapped by `<think>` (token id 12) and `</think>` (token id 13). [S1] [S3]
- **NVIDIA-recommended sampling for reasoning tasks: `temperature=1.0, top_p=1.0`**. For tool-calling: `0.6 / 0.95`. [S1]
- Custom reasoning parser for vLLM: `nano_v3_reasoning_parser.py` (download from the HF repo). Tool-call parser: `qwen3_coder`. [S4]

### Baseline accuracy on NVIDIA-reported benchmarks [S1]
- AIME25 (no tools): **89.1 %**
- GPQA (no tools): **73.0 %**
- MMLU-Pro: **78.3 %**
- LiveCodeBench (2025-05 → 08): **68.3 %**
- MiniF2F pass@1 / pass@32: 50.0 / 79.9

This is already a very strong base. Implication for this competition: the NVIDIA-private benchmark is expected to be *specifically hard for this model*, i.e. out-of-distribution from Nemotron-3-Nano's training mix.

## 2. LoRA-on-Nemotron-3-Nano support status (inference side)

| Question | Answer | Source |
|---|---|---|
| Is LoRA for Nemotron-H merged into vLLM? | **Yes**, PR #30802 by `danisereb@nvidia.com`, labeled `ready`, merged ≈ Jan 2026 | [S5] |
| vLLM version with full support | **vLLM 0.12.0**; 0.11.2 partial; the competition metric uses vLLM with the same config | [S4] |
| How was it tested? | PEFT adapter with **`target_modules="all-linear"`** | [S5] |
| Known exclusion | **`VLLM_USE_FLASHINFER_MOE_FP8` NOT supported** with LoRA at time of PR | [S5] |
| Recommended training frameworks | (a) **Megatron-Bridge** recipes (NVIDIA official) [S6]; (b) **Unsloth day-zero support** [S3] | HF discussion #17 |

### Unsloth-specific guidance for this model [S3]
- 16-bit LoRA fine-tuning of Nemotron-3-Nano 30B ≈ **60 GB VRAM** → fits 80 GB A100 and 96 GB RTX PRO 6000.
- **Disable router-layer training** ("not a good idea to fine-tune the router layer") — Unsloth does this by default.
- **Mix ≥75 % reasoning + ≤25 % non-reasoning** in the training data to retain reasoning ability.
- Chat-template format:
  ```
  <|im_start|>system\n<|im_end|>\n<|im_start|>user\nWhat is 1+1?<|im_end|>\n<|im_start|>assistant\n<think></think>2<|im_end|>\n...
  ```
  Empty `<think></think>` = reasoning off; non-empty = reasoning on.

## 3. Kaggle harness specifics

### Eval config (fixed by competition) [S7]
```
max_lora_rank          = 32
max_tokens             = 7680
top_p                  = 1.0
temperature            = 0.0       # DETERMINISTIC, outside NVIDIA-recommended 1.0
max_num_seqs           = 64
gpu_memory_utilization = 0.85
max_model_len          = 8192
```
Inference engine: **vLLM** with the Nemotron-3-Nano-30B-A3B base + your uploaded LoRA adapter. Hardware: **RTX PRO 6000 (96 GB, Blackwell)** on Google Cloud G4 VMs (approximately).

### Metric logic (confirmed from competitor code matching official notebook) [S8]
```python
# 1. Find \boxed{...} first (regex: r'\\boxed\{([^}]*)(?:\}|$)')
# 2. Fallbacks: "The final answer is:", "Final answer:", "Final answer is:"
# 3. Fallback: last numeric value in text
# 4. Fallback: last non-empty line
# Comparison:
#   - Binary strings (only 0/1)  → case-insensitive string compare
#   - Parseable floats           → math.isclose(rel_tol=1e-2, abs_tol=1e-5)
#   - Otherwise                  → case-insensitive string compare
```
**Known metric bugs:** (a) `\boxed{}` cannot contain a `}` character — extractor stops at first `}`. (b) Eval regex cannot parse answers starting with `}`. (c) Scoring is **not deterministic** (confirmed by Ryan Holbrook) — identical submissions can drift by 0.02–0.04. [S8]

### Baseline demo [S9]
- Notebook: `ryanholbrook/nvidia-nemotron-submission-demo`
- **Public score 0.49–0.50**, runtime **16 m 43 s** on RTX Pro 6000.
- Uses base model + simple boxed prompt. No adapter.

### Training hardware envelope on a single RTX PRO 6000 (96 GB) [S10]
From Tong Hui Kang's compute memo, assuming `target_modules="all-linear"` + LoRA rank 32 + seq len 8192:

| Component | μ=1 | μ=4 | μ=16 |
|---|---|---|---|
| Base BF16 weights | 63.6 GB | 63.6 | 63.6 |
| LoRA weights (FP32) | 3.5 | 3.5 | 3.5 |
| LoRA grads (FP32) | 3.5 | 3.5 | 3.5 |
| Optimizer m+v (FP32) | 7.1 | 7.1 | 7.1 |
| **Total w/ fused CE** | **83.4 GB** | **91.3 GB** | **122.8 GB** (OOM) |

Fits μ=1 comfortably, μ=4 marginally, μ=16 never. Unsloth's **fused cross-entropy** is mandatory for headroom. [S10]

### LoRA trainable-parameter count at rank 32, all-linear [S10]
| Module | Params |
|---|---|
| Attention (q/k/v/o × 6 layers) | 3.74 M |
| Mamba-2 in/out_proj (× 23) | 17.38 M |
| **MoE routed experts (fc1+fc2 × 2944)** | **856.16 M** |
| MoE shared experts (× 23) | 9.42 M (optional) |
| MoE gate router (× 23) | 2.07 M (should skip) |
| lm_head + embed_tokens | 8.56 M |
| **Total** | **≈ 877–897 M trainable params** |

**Implication:** "rank 32 LoRA" on this MoE is actually **~880 M trainable params**, ~97 % of which sit in the routed expert projections. Whether you touch every expert uniformly is a real design choice (see R4 verification log for open question).

### Training throughput on RTX PRO 6000 [S10]
- Forward pass: ~57 TFLOP/sample (3.5 B active × 8192 × 2)
- Forward+Backward w/ checkpointing: ~228 TFLOP/sample
- Theoretical ceiling on RTX PRO 6000 (252 BF16 TFLOPS): **~0.9 s/sample at 100 % MFU**. Realistic MFU 15–25 % → **3–6 s/sample**.
- For 9500 training examples × 1 epoch × μ=1: **8–16 h per epoch**. Several epochs over categories is feasible in 24–72 h of wall-clock.

### GRPO cache-bug (critical)
NVIDIA's `modeling_nemotron_h.py` has a parameter-name mismatch: `prepare_inputs_for_generation` emits `past_key_values` but `forward` accepts `cache_params`. The cache falls into `**kwargs` and is silently ignored, collapsing generation to ~**2 tok/s** vs. a fixed **~38 tok/s** (19× slower). [S11]

- **Affects:** GRPO / any auto-regressive generation in training. **Does not** affect SFT (no generation).
- **Does not** affect vLLM-based eval (vLLM uses its own cache).
- **Fix:** `pip install "transformers>=5.3.0"` and **drop `trust_remote_code=True`** from model loading; also set `gradient_checkpointing=False` for training (NemotronHForCausalLM doesn't declare support). [S11]

## 4. Official NVIDIA tuning resources

1. **Nemotron developer repo** — end-to-end recipes, cookbooks, and CLI: `github.com/NVIDIA-NeMo/Nemotron` with `src/nemotron/recipes/` and `usage-cookbook/`. Includes CLI `nemotron nano3 sft`, `nemotron nano3 rl`, data import/prep commands. [S12]
2. **NeMo-RL guide for Nemotron 3 Nano** — GRPO/RLVR walkthrough at `docs.nvidia.com/nemo/rl/nightly/guides/nemotron-3-nano.html`. [S13]
3. **Nemotron 3 Super GRPO/DAPO notebook** — reusable recipe: `usage-cookbook/Nemotron-3-Super/grpo-dapo`. [S13]
4. **vLLM deployment cookbook** — `NVIDIA-NeMo/Nemotron/usage-cookbook/Nemotron-3-Nano/vllm_cookbook.ipynb`. [S4] [S12]
5. **Megatron-Bridge fine-tuning recipes** — `NVIDIA-NeMo/Megatron-Bridge/docs/models/llm/nemotron3.md#finetuning-recipes`. [S6]
6. **Unsloth notebook** — 80 GB A100 Colab template for 30 B LoRA, with Nemotron-3 day-zero support, and a collaboration with NeMo Gym for single-turn RL rollouts. [S3]
7. **NeMo Gym** — RL environments library. [S14]
8. **Nemotron-Post-Training-v3 datasets** — the SFT/RL data mix used to post-train the released instruct checkpoint, including synthetic LIMO from DeepSeek-R1-0528, synthetic OpenMathReasoning / OpenCodeReasoning from gpt-oss-120b, NuminaMath-CoT, Nemotron-PrismMath. [S1]

## 5. Unknown / unresolved

1. **Exact LoRA target set for rank-32 on MoE:** whether to train router gates and shared experts is not fixed — Unsloth says skip router, NVIDIA PR tests "all-linear," Tong's memo lists both configurations. Ablation needed.
2. **Whether FP8 base is usable under LoRA:** the vLLM PR did not add support for `VLLM_USE_FLASHINFER_MOE_FP8`, so the BF16 base is the safer path for both training and inference. Worth a direct test if model-load time or VRAM becomes a blocker.
3. **Behavior of the custom `nano_v3_reasoning_parser`** when the model is run with T=0 vs. T=1 — parser was shipped for T=1 reasoning workflows but the Kaggle harness hard-sets T=0. The boxed-answer extractor is the primary source of truth; the reasoning parser is not used by the Kaggle metric.
4. **Non-determinism in the scoring server** (confirmed by Ryan Holbrook): not diagnosed publicly. Participants see ±0.01–0.04 drift across resubmissions of identical adapters.

## Sources

[S1] `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` — Hugging Face model card. https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
[S2] NVIDIA (Dec 2025). "Nemotron 3 Nano: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning." arXiv 2512.20848. https://arxiv.org/abs/2512.20848
[S3] Unsloth docs — "NVIDIA Nemotron 3 Nano - How To Run Guide." https://docs.unsloth.ai/models/nemotron-3
[S4] vllm-project/recipes — `NVIDIA/Nemotron-3-Nano-30B-A3B.md`. https://github.com/vllm-project/recipes/blob/main/NVIDIA/Nemotron-3-Nano-30B-A3B.md
[S5] vllm-project/vllm PR #30802 — "Add support for LoRA adapters in Nemotron-H models" (danisereb@nvidia). https://github.com/vllm-project/vllm/pull/30802
[S6] NVIDIA-NeMo/Megatron-Bridge — `docs/models/llm/nemotron3.md#finetuning-recipes`. https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docs/models/llm/nemotron3.md
[S7] Kaggle NVIDIA Nemotron Reasoning Challenge — Overview/Evaluation page (user-provided text + competition overview). https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/overview
[S8] m4nocha (Kaggle), "Is RLVR worth it? or should I work on SFT only?" (contains full extract_final_answer + verify implementation that matches the official metric). https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689840
[S9] Ryan Holbrook, "NVIDIA Nemotron Submission Demo" — Kaggle notebook. https://www.kaggle.com/code/ryanholbrook/nvidia-nemotron-submission-demo
[S10] Tong Hui Kang, "Counting memory and compute" (competitor post with memory/throughput breakdown). https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/687961
[S11] Komil Parmar, "Why GRPO is Painfully Slow on Nemotron (and the Fix)"; Ashutosh Kumar, "BUG in Nemotron Model file://Models/modeling_nemotron_h.py". https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/690161 + .../686615
[S12] NVIDIA-NeMo/Nemotron developer repo README + structure. https://github.com/NVIDIA-NeMo/Nemotron
[S13] NVIDIA NeMo-RL — Nemotron 3 Nano guide. https://docs.nvidia.com/nemo/rl/nightly/guides/nemotron-3-nano.html (linked from Kaggle host resources post https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/681745)
[S14] NVIDIA-NeMo/Gym. https://github.com/NVIDIA-NeMo/Gym
