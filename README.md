# Kaggle Competitions Repository

This repository contains my Kaggle competition submissions, notebooks, and supporting code.

## Competitions

### NVIDIA Nemotron Model Reasoning Challenge
- **Status:** Active
- **Deadline:** 2026-06-15
- **Repository:** [subhopam-das-personal/nvidia-nemotron-ml](https://github.com/subhopam-das-personal/nvidia-nemotron-ml)
- **Score:** TBD

## Repository Structure

```
├── README.md
├── competitions/
│   ├── nvidia-nemotron-reasoning/
│   │   ├── notebooks/           # Kaggle notebooks ready to upload
│   │   ├── data/               # Generated training data
│   │   ├── src/               # Supporting code (oracles, eval harness)
│   │   └── submissions/        # Trained adapters (rank ≤ 32)
└── shared/                 # Shared utilities, scripts
    ├── notebooks/            # Exploratory notebooks
    └── archive/             # Old competition submissions
```

## Quick Start

1. Each competition gets its own folder under `competitions/`
2. Upload notebooks directly from `competitions/[name]/notebooks/` to Kaggle
3. Commit trained adapters to `competitions/[name]/submissions/`
4. Follow the README in each competition folder

## License

MIT License - feel free to use this code for your own competitions.

---

*Last updated: $(date +%Y-%m-%d)
