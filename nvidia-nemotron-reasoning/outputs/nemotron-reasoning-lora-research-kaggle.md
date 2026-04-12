# Research File R4 — Kaggle Competition Intelligence

**Status:** T4 done. This is the highest-signal file for strategy; the competition is far more about task-specific engineering than about generic reasoning-LoRA methodology.

## 1. What the "novel reasoning benchmark" actually looks like

From competitor reverse-engineering [S1] [S2]: the training set has **9,500 problems** that cleanly partition into **six (sometimes called seven) templated categories**. All answers are deterministic, verifiable, and fit the `\boxed{...}` grading contract.

| # | Category | What it is | Reverse-engineering status |
|---|---|---|---|
| 1 | **Gravitational Constant** | Physics word problem, plug-and-chug with g | Fully solvable; competitors hit **100 %** with clean CoT traces [S1] |
| 2 | **Number Base Conversion** | Base-N ↔ decimal, "secret number" wording | **100 %** with template-solver traces [S1] |
| 3 | **Unit Conversion** | Metric/imperial conversions | **100 %** [S1] |
| 4 | **Text Encryption** | Caesar / Vigenère / shift ciphers | **~96 %** [S1] |
| 5 | **Bit Manipulation** | Bitwise ops on 8-bit integers, rotations, per-bit boolean rules | **9–46 %**; model struggles with xnor/or_not/and_not composition [S1] [S3] |
| 6 | **Equation Transformation** | *Meta-puzzle*: infer a bijection {symbols→digits} and {symbols→operators} from few-shot examples, then apply to a query. Includes numeric and **symbolic** sub-cases | Numeric ~50 %, **symbolic ~4 %** [S1] [S3] |

(A "Roman numerals" sub-category was also mentioned but seems to roll under unit/number conversion in some splits. [S2])

### Equation-Transformation is where the leaderboard lives
- Symbols can map to digits 0-9 (bijection) and operators can map to {+, −, ×, ÷, GCD, abs-diff, reverse-then-op, …}. [S3]
- Many problems are genuinely **unsolvable by pattern matching** because the query contains an operator that never appears in the few-shot examples. Even `opus-4.6` fails those. [S3]
- Top teams crack most numeric equation-transformation via careful CoT scaffolds: *"enumerate digit→symbol bijection, enumerate operator hypotheses, verify on examples, apply to query, re-encode."* One example of such a scaffold from a competitor (decoding "`:<-]!`" to "`-^]`") is walked through in [S3].

### Training-data hygiene warnings
- **Bit Manipulation:** 764 / 1513 (50.5 %) of training traces contain a mismatch between the computed "Result" line and the `\boxed{}` answer → **half the official traces teach the wrong reasoning**. [S4]
- **Equation Transformation:** 1555 / 3157 (49 %) training examples have input/output length mismatches that break a naive char-map strategy; a further 1382 (44 %) contain unknown characters; the competition organizers have not indicated a fix. [S4]
- **Implication:** *do not train naively on the provided traces*. Generate fresh traces with a stronger teacher, or filter aggressively with a programmatic verifier.

## 2. Leaderboard snapshot (as of 2026-04-12, ~72 h after the Apr 9 midpoint cutoff)

| Rank | Team | Public score | Entries |
|---|---|---|---|
| 1 | Tong Hui Kang | **0.85** | 40 |
| 2 | Just a test | 0.84 | 35 |
| 3 | toxu | 0.83 | 45 |
| 4 | Alice's Wonderland | 0.82 | 10 |
| 5 | NewMes Team | 0.80 | 9 |
| 6–13 | (gold zone) | **0.79** | ... |
| 14–46 | (silver zone) | 0.76–0.79 | ... |
| Top-10 % ≈ | — | **~0.79** | — |
| **NVIDIA demo baseline** | Ryan Holbrook | **0.49–0.50** | — |

Source: [S5]. **50/50 public/private split** — the private LB may reshuffle substantially. Scoring is non-deterministic: identical adapters drift by 0.01–0.04 per submission. [S4]

### Calibration of category difficulty from one competitor's run (m4nocha, local val) [S1]
```
OVERALL 81.34 %  (693/852)
  Bit Manipulation         :  9.23 %
  Equation Transformation  : 52.11 %
  Gravitational Constant   : 100 %
  Number Base Conversion   : 100 %
  Text Encryption          : 95.86 %
  Unit Conversion          : 100 %
```
With four easy categories pinned at 100 %, the realistic **accuracy ceiling** reported by multiple competitors is **~0.90**. The fact that #1 is at 0.85 and the cluster is at 0.79 suggests that (a) the four easy categories are universally solved, (b) Equation-Transformation-Symbolic is the principal gradient, (c) Bit-Manipulation is the secondary gradient. Attacking these two is where prize money is.

## 3. Public notebook landscape

All Apache-2.0, as of 2026-04-12. [S6]

| Notebook | Score | Approach |
|---|---|---|
| `ryanholbrook/nvidia-nemotron-submission-demo` | 0.50 | Official baseline, no training |
| `rishabhshukla111/notebookneotronreasoningmodel` | 0.53 | Minimal SFT |
| `alexxxsem/nemotron-simple-offline-working-demo` | 0.63 | Base model + prompt engineering |
| `sabreenelkamash/nvidia-nemotron-model-reasoning-challenge` | 0.60 | — |
| `konbu17/doc-to-lora-knowledge-injection-nemotron-3-nano` | 0.60 | Knowledge-injection LoRA |
| `waterjoe/nvidia-nemotron-mask-loss` | 0.66 | SFT + masked loss on traces |
| `pauldumontunc/lora-sfttraining-cot-0-64` | 0.64 | LoRA SFT on CoT |
| `jek1wantaufik/nvidia-nemotron-model-reasoning-0-68` | 0.68 | LoRA SFT + improved CoT |
| `sorajiang/notebooksft` | **0.70** | SFT-only |
| **`amanatar/nemotron-ultimate-sft-grpo-v3`** | **0.70** | **SFT + GRPO**, 7 h 12 m run on RTX Pro 6000, `nemotron-sft-lora-cot-selection` dataset |

Supporting data notebooks: `occultainsights/synthetic-data-generation`, `konbu17/cot-generation-for-nemotron-challenge`, `konbu17/bit-manipulation-solver-cot-generator`, `hoangvux/nvidia-nemotron-synthetic-data`, `pjt222/nemotron-cot-review`.

**Gap:** public notebook ceiling is **0.70** while the leaderboard top is **0.85** — a +0.15 gap that reflects private data/CoT engineering. The Open Progress Prize winner (Tong Hui Kang) committed to publishing their method by Sunday April 12 UTC; as of this scan (~05:00 UTC on that Sunday) the writeup is still a placeholder post only [S7]. Watch [S7] for the imminent full writeup — this is the single highest-leverage external input for the user's strategy.

## 4. What the top of the leaderboard is doing (speculation + direct signals)

From the Open Progress Prize discussion [S7]:
- **Tong Hui Kang** (1st, 0.85) previously worked on **AIMO 3** and **ARC-AGI-2**; has an AIMO corpus visualizer at aimo.huikang.dev. Approach is Kaggle-AIMO-style: aggressive data generation + verifiability. GitHub repo for this comp: `github.com/tonghuikang/nemotron` (data visualizer with per-problem jsonl files).
- **Guesses by other competitors** about what the secret is, with supporting signals:
  - **Russell Kirk (45th):** "the 'secret' should be how to make steps tractable and in the correct order if you're doing SFT. RLHF lets the model figure that out." → structured decomposition of reasoning steps.
  - **Svanik Kolli (349th):** "Rejection Sampling with a Verifiable Reward (RLVR) → force the model to generate specific reasoning thought blocks that must lead to a correct symbolic answer before training on them."
  - **James Day (83rd):** local ensemble ceiling ~0.90; the remaining ~0.10 is "probably faulty samples."
  - **Komil Parmar (338th):** "the secret ingredient (if any) would be the data, or the teacher model"; estimates ~$25–$100 API spend is sufficient.
- **Spend estimates:** Tong reported running on Modal RTX PRO 6000 at **$3.03/h** [S2]; Tito_42 rents GCP spot RTX PRO 6000 at **$0.90/h** [S2]. Competitor estimates of total spend for the midpoint winner: **$100 – $500**, very plausibly under $100 for compute if spot GPUs are used efficiently.
- **Tong's own compute signal:** inferring once over all 9500 problems with the base model generated 48.2 M tokens at 2.5k tok/s with `--max-num-seqs 256` → **≈ 5.4 hours ≈ $4.50** on a GCP spot RTX PRO 6000. Full category-specific trace generation with self-verification is an order of magnitude beyond that but still in **tens of dollars**. [S2]

### Data-generation playbook visible in public discussions
1. **Reverse-engineer the generator.** For 5 of 6 categories, the transformation is algorithmic and can be written as deterministic Python code. Competitors are openly sharing partial solvers. [S1] [S3]
2. **Write a big teacher solver / use a frontier LLM with a detailed decoding prompt.** Working examples in [S3] show Gemini producing correct digit+operator bijections after careful scaffolding.
3. **Verify** each generated trace programmatically (substitute back, check all few-shot examples, confirm the final answer) and keep only the ones that match.
4. **Distill** verified traces into the Nemotron chat format with `<think>...</think>` blocks and a terminal `\boxed{answer}`.
5. **SFT the LoRA** on the cleaned traces. Optional: follow with short GRPO on hard categories using the programmatic verifier as reward.

## 5. Rule / IP constraints that actually matter

| Rule | Practical implication | Source |
|---|---|---|
| **Teacher model for distillation**: "distilling Gemini Flash 2.0 outputs looks fine for this competition" (CPMP, host, Apr 2026) | DeepSeek-R1 (MIT) is cleanest; Gemini Flash 2.0 and GPT-OSS 20B/120B (Apache 2.0) explicitly OK for this comp; Anthropic Claude is gray — host allows it but Anthropic ToS prohibits training competing models. For Open Contribution/methodology prizes, **use permissively-licensed teachers** to avoid disqualification. | [S8] |
| **Prize eligibility** requires a public Kaggle notebook + write-up by the *relevant* deadline. Open Progress Prize: public by Apr 9 (passed). Methodology cutoff: Apr 16, 2026. Final: Jun 15, 2026. | Budget writing + documentation time, not just training. | [S9] |
| **Hosts can remove from LB** (not just deny prize) for rule violations — confirmed by Christof Henkel. | Don't cut corners on licensing; cite your datasets. | [S8] |
| **"Reasonably accessible to all" and "minimal cost"** rule on external resources/APIs | Ambiguous, but competitors interpret ~$50 as clearly fine, and a $500 run is on the edge. Making the teacher data public (e.g. as a Kaggle dataset) is cheap insurance for methodology prizes. | [S8] |
| **LoRA rank ≤ 32**, submission.zip must include `adapter_config.json` | Fixed. Use PEFT's default save; include `base_model_name_or_path` and all target modules. | [S9] |
| **Deterministic eval harness (temperature=0)** means **no self-consistency / majority voting** is possible at inference time. | All accuracy gain must come from the *single* generation. | [S9] |

## 6. Format engineering cheat-sheet

Derived from metric source + competitor findings [S4]:

- **Always end with `\boxed{answer}`** as the last substring in the trace. The extractor falls back to heuristics when `\boxed{}` is absent, which is unreliable.
- The regex is `r'\\boxed\{([^}]*)(?:\}|$)'` — it matches everything up to the **first** `}`. **Do not put `}` inside the answer.** If your LoRA starts producing set notation or `f(x)} = ...` it will silently truncate. [S4]
- For binary answers (bitstrings) the comparator is **strict string match** — make sure the LoRA produces the exact expected length and no leading/trailing whitespace or `0b` prefix.
- For numeric answers, tolerance is `rel_tol=1e-2, abs_tol=1e-5`. A rounded answer at 3 sig figs is always safe.
- With `max_tokens=7680` and `max_model_len=8192`, the prompt itself consumes the slack — keep reasoning traces ≤ ~7 k tokens and always leave room for the `\boxed{}` tail. Truncation before the boxed tag = zero credit.
- Use `<think>...</think>` blocks for reasoning; the Nemotron chat template supports it natively (token IDs 12 and 13). The Kaggle metric **ignores** the `<think>` content and grades only what comes after. [S4]
- Tong Hui Kang's base-model inference uses `temperature=1.0` for data generation (to see log-probs distribution), not eval. [S2] For *eval-mode generation during training*, run at `T=0` so your local validation matches the leaderboard.

## 7. Current blockers / open questions for the user

1. **Tong's imminent writeup** (promised by 2026-04-12 UTC). The methodology for the 0.85 submission will be shared at `https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915` with notebooks, data, and a visual interface. **This should be the user's first read on the morning of April 12.** [S7]
2. **Does anyone share a cleaned version of the 9500-problem training set** with corrected bit-manipulation and equation-transformation traces? The `kienngx/nemotron-30b-competition-trainingdata-cot-labels` dataset exists [S8] but has Gemini-2.0-flash traces (host-cleared for this comp, but check).
3. **Is the private test set drawn from the same 6 categories?** This is the big unknown — if NVIDIA Research holds back a 7th category, category-specific overfitting will regress on private.

## Sources

[S1] m4nocha (aksh1t), "Is RLVR worth it? or should I work on SFT only?" — contains per-category accuracy breakdown for SFT-only and the `extract_final_answer`/`verify` regex. https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689840
[S2] Tong Hui Kang (huikang), "Visualize the problems and completions from the base model" + companion post at nemotron.huikang.dev. https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/684212
[S3] Dennis (dennisfong), "[Dataset Hallucination?] How did you resolve these problems by human?" — symbolic equation-transformation walkthroughs by dangnh0611 and others. https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/684192
[S4] Komil Parmar, "Why GRPO is Painfully Slow on Nemotron (and the Fix)"; Ashutosh Kumar, "How to Get Started …" thread on training-data bugs; "Edge case in metric: \boxed{} cannot contain }". https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/690161 , .../681745 , .../689257
[S5] Leaderboard snapshot retrieved 2026-04-12 at ~04:00 UTC. https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/leaderboard
[S6] Code tab — public notebooks, retrieved 2026-04-12. https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/code
[S7] Tong Hui Kang, "[Open Progress Prize Publication] Placeholder post." https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915
[S8] c-number, "Do not distill models that do not allow distillation …" with CPMP (host) clearance for this competition. https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/688360 ; lakshmig82, "Open Progress Prize question." https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/685031
[S9] NVIDIA Nemotron Model Reasoning Challenge — Overview / Rules. https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/overview
