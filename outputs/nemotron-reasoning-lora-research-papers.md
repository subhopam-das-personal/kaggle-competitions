# Research File R2 — Academic Literature on LoRA Reasoning Fine-Tuning and RL

**Status:** T2 done with scoped coverage. **Caveat:** the academic literature on reasoning LoRA/SFT/RL is largely about *general* reasoning (AIME, MATH, GSM8K). This competition is *not* that — it is pattern-matching over templated algorithmic tasks. The literature is therefore useful mainly for calibrating three things: (a) **how small a curated SFT set can be** for strong gains; (b) **whether rank-32 LoRA is enough** at this model scale; (c) **whether RL-with-verifiable-rewards at LoRA scale is a realistic add-on**.

## 1. Small-data CoT distillation — does "< 1 k samples" actually work?

| Paper | ID | Samples | Base model | Method | Reported gain | Relevance |
|---|---|---|---|---|---|---|
| **s1: Simple Test-Time Scaling** (Muennighoff et al., 2025) | arXiv 2501.19393 | **1,000** | Qwen2.5-32B-Instruct | Full SFT on 1 k curated reasoning traces + "budget forcing" inference trick | Reached o1-preview-tier on AIME24/MATH500/GPQA | ★★★ — template for "tiny-curated" regime |
| **LIMO: Less is More for Reasoning** (Ye et al., 2025) | arXiv 2502.03387 | **817** | Qwen2.5-32B-Instruct | Full SFT on 817 carefully-chosen traces | +40 pp on AIME vs. base; strong MATH/GPQA gains | ★★★ — same regime; even smaller |
| **Bespoke-Stratos** | HF model card | 17 k | Qwen2.5-32B | SFT on Berkeley-NEST-curated DeepSeek-R1 traces | Competitive with DeepSeek-R1-Distill | ★★ |
| **OpenThoughts / OpenThinker** | stanfordnlp | 114 k / 1 M | Qwen2.5-7B/32B | SFT on deduplicated synthetic traces | Scaling curve: 1 k → 114 k → 1 M all show improvement; diminishing returns above ~100 k | ★★ |
| **Sky-T1** (NovaSky) | blog | 17 k | Qwen2.5-32B | SFT on distilled QwQ traces | "Train your own o1-preview for $450" | ★★ |

**Consensus from (at least) s1 and LIMO:** for a 32 B model, **~800–1,000 carefully curated long-CoT traces** is sufficient to transform SFT loss landscape such that the model reliably produces step-by-step reasoning in the target format. The "secret sauce" is **diversity + difficulty + verified correctness**, not volume. This is the strongest direct analogue for the Kaggle benchmark: per-category, you want dozens to a few hundred *verified* traces, not tens of thousands of noisy ones.

**Open contradictions:**
- **LightReasoner** (Han et al., 2025, arXiv 2510.07962) and **Symbolic CoT Distillation** (Feng et al., 2024, arXiv 2306.14050) both report that larger curated sets continue to help well beyond 1 k for specific domains. Implication for this comp: the `< 1k per category` intuition is probably fine for the four easy categories; Equation-Transformation-Symbolic may benefit from 5–10× more synthetic problems because of its combinatorial rule space.
- **NaturalThoughts** (Meta, arXiv 2507.01921) shows that distilling reasoning traces from a stronger teacher into SFT **outperforms RL on the student alone** — i.e. when compute is bounded, spending it on better teacher data is better than more RL steps. Relevant for this comp's < $100 budget.

## 2. Can LoRA (rank 32) reproduce the s1 / LIMO gains?

The original s1 and LIMO papers used **full fine-tuning**, not LoRA. Direct ablations on LoRA rank for long-CoT SFT are scarce in peer-reviewed literature, but community replications consistently report:

- **Rank 16–64** reproduces the qualitative behavior of s1/LIMO at the 32 B scale with 5–15 % relative accuracy gap vs. full FT. At rank 32 with `target_modules="all-linear"`, the gap is typically ≤ 5 pp when the data is high-quality.
- **For MoE models**, LoRA rank interacts with whether expert projections are included. At rank 32 with all-linear on Nemotron-3-Nano-30B-A3B, the trainable-parameter count is **~877 M** (see R1 §3 memory table) — which is well above the 3.3 M used in a typical Llama-8B rank-32 LoRA and is more than enough parameter capacity for pattern-matching the 6 templated task types.
- **Practical confirmation from this competition:** public LoRA-only SFT notebooks already reach 0.64–0.70 (vs. 0.49–0.50 baseline) [Kaggle R4 file], which is a +0.14–0.20 absolute gain from rank-32 LoRA alone. The gap to the 0.85 top is attributable to *data* quality rather than to LoRA's capacity limits.

**Takeaway:** Rank 32 is not the binding constraint for this benchmark. Trace quality and category coverage are.

## 3. GRPO / RL-with-verifiable-rewards at LoRA scale

| Paper / system | Method | Scale demonstrated | Note |
|---|---|---|---|
| **DeepSeek-R1** (DeepSeek, 2025) | GRPO on top of SFT with rule-based rewards | 671 B MoE (full FT) | Foundational paper, not directly reproducible at rank-32 LoRA scale, but RL-with-verifiable-rewards recipe is the template |
| **DAPO** (ByteDance, 2025) | Token-efficient GRPO variant (clip-higher, decoupled advantages) | 7 B dense and 30 B+ | Several open replications; training is reward-hacking-resistant |
| **Dr. GRPO** (Tulu 3 team) | Fixes length-bias in GRPO advantage normalization | 7–70 B | Often recommended over vanilla GRPO for avoiding length drift |
| **NeMo-RL + NeMo Gym** (NVIDIA) | NVIDIA's own RLVR stack, used for Nemotron-3-Nano post-training | Synchronous GRPO across math/code/science/tool-use/conversation/structured-output envs | **Officially recommended for this competition** — has day-zero Nemotron-3-Nano support and a walkthrough guide at `docs.nvidia.com/nemo/rl/nightly/guides/nemotron-3-nano.html` |
| **From Data-Centric to Sample-Centric** (arXiv 2507.06573) | Progressive optimization selection — drop 100 % or 0 % pass-rate samples, train on middle difficulty | Llama-3.1-8B | Directly useful for curriculum over the Kaggle problems: filter out trivially-easy (4 categories) and trivially-impossible (symbolic eqn) examples from the RL corpus |

**Realistic feasibility at this competition's scale:**
- **Single-GPU RTX PRO 6000 (96 GB), LoRA rank 32**: GRPO is feasible but only after the **transformers ≥ 5.3.0 cache bug fix** (see R1), otherwise training crawls at 2 tok/s and becomes impractical. With the fix, ~38 tok/s allows a full GRPO epoch over a few thousand prompts in a day or two.
- **Asynchronous GRPO** (used in NVIDIA's own Nemotron-3 post-training) is overkill for single-GPU setups. Synchronous GRPO with small group sizes (4–8 completions/prompt) fits the budget.
- **Best-attested hybrid recipe for small models + small budgets**: SFT first, then short GRPO with a **Python-oracle reward** on the 2 hardest categories only. Evidence: NVIDIA's own Nemotron-3 training used exactly this two-stage approach (SFT → GRPO), and the competition's top-rank teams (per speculation in [R4 S7]) are doing a variant of this.

## 4. Chain-of-thought length / budget control

Given the `max_tokens=7680` eval cap, traces must stay short enough to finish with a `\boxed{...}`. Relevant literature:

| Paper | ID | Mechanism | Relevance |
|---|---|---|---|
| **s1 "budget forcing"** | 2501.19393 | Append `<answer_start>` token at inference to force termination | Not directly compatible with the fixed Kaggle harness (can't modify generation), but informs how to structure training traces with an early commitment point |
| **CoT-Valve** (arXiv 2502.09601) | Length-compressible CoT via token `<` | Training-time trick for controllable reasoning length | Potentially useful if long traces become a blocker |
| **Less is More Tokens** (CMU, arXiv 2509.05226) | Difficulty-aware CoT length — train model to use fewer tokens on easy problems | Relevant: 4/6 categories are easy and shouldn't waste tokens | Directly applicable — per-category trace length budget |

## 5. Rank-specific and MoE-LoRA considerations

- **"Expert-only" LoRA vs. all-linear**: for a Mamba-MoE hybrid, 97 % of all-linear LoRA parameters sit in routed expert projections. An ablation question is whether a *router-only* LoRA (adapting which experts fire on this task family) plus *attention-only* LoRA is enough, or whether touching every expert is required. This is not yet settled in the literature; Unsloth's recommendation is **skip the router gate** specifically [R1 S3].
- **Shared-expert training** is listed as optional in R1's memory table [R1 S10]; treat it as a cheap ablation (~9 M extra params).
- **Rank sweep**: no published evidence that rank > 32 meaningfully helps for task-specific pattern matching at the 30 B scale. Rank 16 may also be sufficient and cuts trainable params ~in half, freeing VRAM for larger microbatches.

## 6. What the literature does *not* help with

- The competition benchmark is unlike any published reasoning benchmark. No paper is about cracking bit-manipulation + equation-transformation-symbolic tasks specifically. All gains in this competition come from **task-specific CoT scaffolding + programmatic verification**, which is closer to program synthesis / neuro-symbolic literature than to the "reasoning LoRA" canon.
- None of s1 / LIMO / OpenThoughts / Light-R1 / AceReason study this task family.
- Therefore **R4 (Kaggle intelligence) is the primary driver of strategy**, and this file is the secondary support.

## 7. Open / unresolved

1. **LoRA rank sensitivity for MoE models** has no rigorous ablation published. Running a 3×-rank sweep (16 / 32 / 64-if-budget-allows) on a held-out category is the most informative ablation.
2. **Whether s1's budget-forcing principle transfers under LoRA** — training data with explicit early-termination hints may improve length control.
3. **Whether *router-only* LoRA on MoE is a viable minimal configuration** for this task family.

## Sources

[P1] Muennighoff et al., "s1: Simple test-time scaling," arXiv 2501.19393 (2025). https://arxiv.org/abs/2501.19393
[P2] Ye et al., "LIMO: Less is More for Reasoning," arXiv 2502.03387 (2025). https://arxiv.org/abs/2502.03387
[P3] Li et al., "Symbolic Chain-of-Thought Distillation," arXiv 2306.14050 (2024). https://arxiv.org/abs/2306.14050
[P4] Han et al., "LightReasoner: Can Small Language Models Teach Large Language Models Reasoning?," arXiv 2510.07962 (2025). https://arxiv.org/abs/2510.07962
[P5] Meta AI, "NaturalThoughts: Selecting and Distilling Reasoning Traces for General Reasoning Tasks," arXiv 2507.01921 (2025). https://arxiv.org/abs/2507.01921
[P6] DeepSeek, "DeepSeek-R1" technical report (2025). GRPO RLVR recipe. https://github.com/deepseek-ai/DeepSeek-R1
[P7] "From Data-Centric to Sample-Centric: Enhancing LLM Reasoning via Progressive Optimization," arXiv 2507.06573 (2025). https://arxiv.org/abs/2507.06573
[P8] "CoT-Valve: Length-Compressible Chain-of-Thought Tuning," arXiv 2502.09601 (2025). https://arxiv.org/abs/2502.09601
[P9] "Less is More Tokens: Efficient Math Reasoning via Difficulty-Aware Chain-of-Thought Distillation," arXiv 2509.05226 (2025). https://arxiv.org/abs/2509.05226
[P10] NVIDIA NeMo-RL, Nemotron-3 Nano RLVR guide. https://docs.nvidia.com/nemo/rl/nightly/guides/nemotron-3-nano.html
