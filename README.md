# Dry Eye NAIST Project

This workspace has been reorganized to separate research, data, code, reports, and plans for a cleaner workflow.

## Directory structure

- `docs/`
  - Research notes and survey summaries in markdown.
- `data/`
  - `surveys/`: survey spreadsheets and dataset metadata.
- `scripts/`
  - Python scripts for dataset search, analysis, and parsing.
- `reports/`
  - `weekly/`: meeting and weekly report files.
- `resources/`
  - `presentations/`: slide decks.
  - `archives/`: ZIP archives and download artifacts.
  - `images/`: visual assets.
  - `unknown/`: nonstandard or incomplete downloads.
- `plans/`
  - `meibomianGlandImprovment/`: MGD-1k segmentation code, results, and experiment notes.
- `.venv/`
  - Python virtual environment (unchanged).

## Primary goal

The current strategic focus is improving meibomian gland segmentation on MGD-1k in `plans/meibomianGlandImprovment/`.

## Next steps

1. Use `plans/meibomianGlandImprovment/train.py` for gland or eyelid training.
2. Use `plans/meibomianGlandImprovment/evaluate_dual_pipeline.py` for the eyelid ROI + gland pipeline.
3. Keep local datasets, server outputs, and presentation assets out of Git.
