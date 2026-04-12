# Research Plan: LoRA Fine-Tuning Strategies for NVIDIA Nemotron-3-Nano-30B on the Kaggle Reasoning Challenge

**Slug:** `nemotron-reasoning-lora`
**Context:** Kaggle "NVIDIA Nemotron Model Reasoning Challenge" (Mar 16 – Jun 15, 2026). Competitor must ship a LoRA adapter (rank ≤ 32, `max_tokens=7680`, `temperature=0.0`, vLLM, `\boxed{}` answer format). Midpoint cutoff Apr 9 already passed; Open Progress Prize methodology cutoff Apr 16, 2026; final deadline Jun 15. The user has 5 submissions/day.

## Objective
Produce an evidence-backed research brief identifying the *highest-expected-value* strategies the user can execute over the remaining ~9 weeks to improve reasoning accuracy on this specific benchmark, given the rank-32 LoRA constraint and vLLM inference setup.

## Questions
1. **Base model reality check.** What exactly is "Nemotron-3-Nano-30B"? Architecture (dense vs. MoE vs. hybrid Mamba), tokenizer, chat template, thinking/reasoning mode toggles, supported LoRA target modules, known quirks under vLLM with LoRA? Is there a reasoning "thinking mode" that must be enabled via system prompt?
2. **Official NVIDIA recipes.** What recipes has NVIDIA published (NeMo-Aligner, NeMo-RL, Nemotron post-training cookbook) that are directly reusable for this model? Any starter notebooks or Kaggle demos?
3. **Best SFT-style reasoning distillation recipes.** What does the recent literature say about making small(-ish) LoRA adapters strong at math/code reasoning via distillation of long chain-of-thought traces? Key papers: s1 / s1.1, LIMO, OpenThoughts/OpenThinker, Sky-T1, Bespoke-Stratos, Light-R1, AceReason-Nemotron. What sample counts, sequence lengths, learning rates, and rank choices actually worked?
4. **RL with verifiable rewards at LoRA scale.** Does GRPO / RLOO / DAPO / RLVR add meaningful accuracy on top of SFT at rank ≤ 32 for a 30B model, given the compute and time budget? What is the smallest credible RL setup reported in the literature? Known instabilities when combining LoRA + RL + long CoT?
5. **Data: the actual datasets people use.** Which open reasoning corpora are: (a) permissively licensed, (b) small enough to train a rank-32 LoRA quickly, (c) empirically effective? OpenMathReasoning, OpenCodeReasoning, Nemotron-Post-Training-Dataset-v1/v2, NuminaMath, DeepSeek-R1 distillation traces, AIME/MATH, OpenR1-Math, AceReason data. How should they be filtered / decontaminated vs. the test set style?
6. **Kaggle-specific evidence.** AIMO-1 and AIMO-2 winning solutions — what generalizes? Any public notebooks, discussion posts, or leaderboard hints for *this* competition? What score range separates baseline / progress prize zone / top-10%?
7. **Inference-format engineering.** Given `temperature=0`, `max_tokens=7680`, single-shot, `\boxed{}` extraction — what prompting and output-format tricks (system prompt, thinking budget, self-consistency, answer reminder) are compatible with this evaluation harness? Are there known LoRA-only behaviors (e.g., forgetting the boxed format) to guard against?
8. **Pitfalls & reproducibility constraints.** Rank-32 cap, vLLM LoRA loading constraints, chat-template mismatches, tokenizer drift, adapter_config.json schema, contamination risk against NVIDIA's private test set.

## Strategy
- **Round 1:** 4 parallel `researcher` subagents on disjoint dimensions (model/docs, reasoning-LoRA papers, datasets, Kaggle/competition intelligence). Each writes its own `outputs/nemotron-reasoning-lora-research-*.md` file.
- **Round 2 (conditional):** Targeted follow-ups only on gaps / contradictions identified after Round 1 (likely: concrete hyperparameters + RL-at-LoRA feasibility).
- **Lead synthesizes** a prioritized strategy brief ranking techniques by expected accuracy gain vs. engineering cost vs. time-to-run on a single G4 (RTX PRO 6000 Blackwell) or similar GPU.
- **Verifier** adds inline citations; **Reviewer** pressure-tests confidence calibration.

### Researcher Allocations
| ID | Dimension | Primary tools |
|---|---|---|
| R1 | Nemotron-3-Nano-30B model card, official NVIDIA recipes, vLLM+LoRA compatibility, Kaggle competition mechanics (scoring code, submission demo) | `web_search`, `fetch_content`, `code_search` |
| R2 | Academic literature on LoRA reasoning distillation + RLVR (s1, LIMO, OpenThoughts, Light-R1, AceReason, DAPO, GRPO-at-scale) | `alpha_search`, `alpha_ask_paper`, `web_search` |
| R3 | Open reasoning datasets: provenance, licensing, size, empirical effectiveness, overlap/contamination concerns | `web_search`, `fetch_content`, `alpha_search` |
| R4 | Kaggle intelligence: AIMO-1/2 winners, public notebooks and discussions for this specific competition, leaderboard signals, known tricks for `\boxed{}` grading | `web_search`, `fetch_content` |

## Acceptance Criteria
- [ ] Every key question answered with ≥ 2 independent sources (or explicitly marked single-source / unknown)
- [ ] A ranked shortlist of 3–5 concrete recipes the user could run, each with: expected gain band, data, hyperparameters, compute estimate, risks
- [ ] Contradictions between sources (e.g., LIMO's "less is more" vs. large-dataset distillation) explicitly reconciled or flagged
- [ ] Confirmation that the recommended recipe is *actually* compatible with rank-32, vLLM loading, and the `adapter_config.json` requirement
- [ ] No single-source claims on critical numbers (accuracy deltas, sample counts, hyperparameters)
- [ ] Decontamination / test-set leakage addressed

## Task Ledger
| ID | Owner | Task | Status | Output |
|---|---|---|---|---|
| T1 | lead (inline) | Model, vLLM LoRA support, Kaggle harness, memory/throughput | **done** | `outputs/nemotron-reasoning-lora-research-model.md` |
| T2 | lead (inline) | Reasoning LoRA/SFT/RL literature + rank/MoE considerations | **done (scoped)** | `outputs/nemotron-reasoning-lora-research-papers.md` |
| T3 | lead (inline) | Dataset options + data-generation pipeline, licensing | **done** | `outputs/nemotron-reasoning-lora-research-data.md` |
| T4 | lead (inline) | Kaggle intel: LB, notebooks, bugs, rules, top-team signals | **done** | `outputs/nemotron-reasoning-lora-research-kaggle.md` |
| T5 | lead | Ranked recipe shortlist + risks | in-progress | `outputs/.drafts/nemotron-reasoning-lora-draft.md` |
| T6 | verifier | Inline citations, URL verification | blocked (subagent API) — lead handled citations inline | `outputs/nemotron-reasoning-lora.md` |
| T7 | reviewer | Confidence/evidence audit | blocked (subagent API) — lead performs claim sweep inline | in draft |

## Verification Log
| Item | Method | Status | Evidence |
|---|---|---|---|
| Canonical model ID is `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (hybrid Mamba-MoE, 30 B total / 3.5 B active) | direct fetch of HF model card | **verified** | [R1 S1] https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 |
| vLLM supports LoRA on Nemotron-H via PR #30802, tested with PEFT `target_modules="all-linear"` | direct fetch of GitHub PR + vLLM recipes page | **verified** | [R1 S5, S4] https://github.com/vllm-project/vllm/pull/30802 |
| Kaggle harness: `max_lora_rank=32, max_tokens=7680, T=0, max_model_len=8192` | user-supplied overview + competitor notebook matching official metric | **verified** | [R1 S7, S8] user msg + https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689840 |
| Baseline demo score 0.49–0.50; leaderboard top 0.85; public notebook ceiling ~0.70 | direct fetch of notebook + leaderboard + code tabs | **verified** (snapshot 2026-04-12) | [R4 S5, S6, S9] |
| Training-data bugs: 50 % Bit-Manipulation and 49 % Equation-Transformation traces mis-aligned | competitor audit post | **verified** (single source, competitor) | [R4 S4] https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/681745 |
| GRPO 2 → 38 tok/s bug and `transformers ≥ 5.3.0` fix | Parmar writeup + CPMP confirmation + Kumar's patch diagnosis | **verified** (multiple independent competitors + host ack) | [R1 S11] |
| LoRA at rank 32 w/ all-linear on this MoE = ~877 M trainable params; μ=1 fits RTX PRO 6000 w/ fused CE at 83.4 GB | Tong Hui Kang memo | **partially verified** (single competitor calc, arithmetic cross-checks) | [R1 S10] |
| Task taxonomy: 6 categories (Grav, Base, Unit, Encryption, BitManip, EqnTransform); 9,500 training problems | Tong visualizer + m4nocha category breakdown | **verified** (two independent competitors) | [R4 S1, S2] |
| Small-data SFT (800–1,000 traces) can deliver large reasoning gains at 30 B scale | s1 and LIMO primary sources | **verified** (two peer papers) | [R2 P1, P2] |
| Host (CPMP) clearance for Gemini 2.0 Flash distillation in this comp only | host post on discussion 688360 | **verified** | [R4 S8] |
| Scoring non-determinism acknowledged by host (Ryan Holbrook) | discussion 687740 | **verified** | [R4 S4] |

## Decision Log
- **2026-04-12 (early):** Plan drafted after clarification. User confirmed target is Kaggle Nemotron-3-Nano-30B reasoning challenge with ~9 weeks remaining and 5 daily submissions. Research-first (strategy brief) before any training code.
- **2026-04-12 (mid):** Subagent-based researcher fan-out failed at the Anthropic API gateway (third-party app credit gating). User chose Option B — inline execution by the lead. All four research files produced in a single session with direct-source fetches.
- **2026-04-12 (late):** Reframing confirmed during research: the "reasoning" benchmark is a set of 6 templated algorithmic categories, not general math. Primary strategy driver is R4 (Kaggle intel), not R2 (literature). Generic open reasoning corpora (OpenMathReasoning, OpenR1-Math, s1K, LIMO) are *methodologically* relevant but *not directly* usable. Focus is on (a) clean re-solving of the 9,500 problems with Python oracles + a teacher LLM, (b) per-category SFT, (c) optional GRPO on the 2 hard categories only. Tong Hui Kang's writeup (promised for Apr 12 UTC) is the single highest-leverage external input remaining.
