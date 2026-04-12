# Provenance: NVIDIA Nemotron Reasoning Challenge — LoRA Strategy Brief

- **Date:** 2026-04-12
- **Slug:** `nemotron-reasoning-lora`
- **Topic:** Strategy brief for a rank-32 LoRA adapter on Nemotron-3-Nano-30B-A3B for the Kaggle NVIDIA Nemotron Model Reasoning Challenge (closes 2026-06-15)
- **Rounds:** 1 research round (inline, single-session, no parallel fan-out)
- **Execution mode:** Lead-only inline execution. The `subagent` tool failed at the Anthropic API gateway with a billing-credit error, so the planned 4-researcher parallel sweep was replaced by sequential inline research using `web_search`, `fetch_content`, `alpha_search`, `get_search_content`. The user explicitly chose this fallback (Option B in prior turn).
- **Unique sources consulted:** 28 (11 Kaggle discussion/leaderboard/code pages, 6 NVIDIA/HuggingFace/vLLM/Unsloth documentation pages, 1 arXiv paper metadata, 2 competitor GitHub repos, 2 alpha paper search results, plus several cross-references within those pages)
- **Sources accepted in brief:** 17 direct citations (labels S1–S17 + P1–P4)
- **Sources rejected / deferred:** Tong Hui Kang's full Open Progress Prize writeup (promised but not yet published as of scan time); detailed contents of the private dataset; public Kaggle notebooks behind a login wall were accessed only via their public preview metadata.
- **Verification:** **PASS WITH NOTES.** The qualitative claims are all backed by ≥1 verifiable primary source; most critical claims (baseline score, leaderboard snapshot, 6 task categories, GRPO cache bug, rank-32 memory arithmetic, dataset bugs) are backed by ≥2 independent sources or direct primary reads. The most important limitations: (a) no empirical runs — all score bands are forward-looking estimates calibrated to public anchors; (b) Tong's method is still speculative pending the imminent writeup; (c) single-source claims where flagged: Tong's memory/compute memo (one competitor's arithmetic), the Ashutosh Kumar training-data-bug audit (one competitor's audit, but methodology described and reproducible).
- **Plan:** `outputs/.plans/nemotron-reasoning-lora.md`
- **Research files:**
  - `outputs/nemotron-reasoning-lora-research-model.md` — base model, vLLM LoRA, Kaggle harness, memory arithmetic
  - `outputs/nemotron-reasoning-lora-research-kaggle.md` — competition intel, leaderboard, public notebooks, rules, known bugs
  - `outputs/nemotron-reasoning-lora-research-data.md` — dataset & data-generation strategy
  - `outputs/nemotron-reasoning-lora-research-papers.md` — literature calibration (s1, LIMO, GRPO, NaturalThoughts, etc.)
- **Canonical deliverable:** `outputs/nemotron-reasoning-lora.md`
- **Workflow deviations from the standard deep-research pattern:**
  - No `verifier` subagent pass — lead performed the citation anchoring inline.
  - No `reviewer` subagent pass — lead performed a claim-sweep table (§4 of the brief) with explicit confidence calibration.
  - No separate `outputs/.drafts/<slug>-draft.md` file — single canonical artifact produced directly as `outputs/nemotron-reasoning-lora.md`.
- **Known MAJOR open items (flagged in §6 of the brief):**
  1. Is the private test set drawn from the same 6 categories? Unknown.
  2. Does router-gate training help on MoE+LoRA? Ablation not run.
  3. Does Tong Hui Kang's writeup change the shortlist materially? Pending publication.
  4. Is the `kienngx` Gemini-labelled CoT dataset clean enough to skip own teacher run? Not audited.
- **Recommended next action for the user:** Check https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915 for Tong Hui Kang's writeup before writing any training code; set up the environment fixes in §5 of the brief; then execute Recipe A.
