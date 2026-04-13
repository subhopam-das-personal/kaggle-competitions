# NVIDIA Nemotron ML Competition

This project is for the NVIDIA Nemotron Model Reasoning Challenge on Kaggle.

## Project Structure

- `nvidia-nemotron-reasoning/` - Main competition code
  - `notebooks/` - Training and evaluation notebooks
  - `scripts/` - Utility scripts for data generation and submission
  - `src/` - Core oracles, data pipeline, and evaluation harness
  - `data/` - Training and test data

## Skill routing

When the user's request matches an available skill, ALWAYS invoke it using the Skill
tool as your FIRST action. Do NOT answer directly, do NOT use other tools first.
The skill has specialized workflows that produce better results than ad-hoc answers.

Key routing rules:
- Product ideas, "is this worth building", brainstorming → invoke office-hours
- Bugs, errors, "why is this broken", 500 errors → invoke investigate
- Ship, deploy, push, create PR → invoke ship
- QA, test the site, find bugs → invoke qa
- Code review, check my diff → invoke review
- Update docs after shipping → invoke document-release
- Weekly retro → invoke retro
- Design system, brand → invoke design-consultation
- Visual audit, design polish → invoke design-review
- Architecture review → invoke plan-eng-review
- Save progress, checkpoint, resume → invoke checkpoint
- Code quality, health check → invoke health
