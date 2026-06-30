# MGD-1k Meibomian Gland Segmentation

This folder contains the training and evaluation pipeline for improving MGD-1k meibomian gland segmentation.

## Purpose
- Organize dataset audit, experiment tracking, model checkpoints, and results.
- Keep the strategy and execution isolated from general workspace files.

## Structure
- `data/`: raw, processed, augmented, and external data assets.
- `experiments/`: separate experiment tracks for baseline, augmentation, transfer learning, and architecture changes.
- `models/`: saved checkpoints and model configurations.
- `results/`: metrics, figures, and outcome comparisons.
- `scripts/`: plan-specific scripts for data inventory, training, and evaluation.
- `docs/`: detailed notes, dataset audit, and strategy refinements.
- `reports/`: weekly updates, summaries, and progress notes.
- `logs/`: training logs and experiment outputs.

## Reproducible Runs

The current baseline behavior is preserved by default. Negative learning is only enabled when `--negative-weight` is greater than `0`.

Baseline gland run:

```bash
python plans/meibomianGlandImprovment/train.py --mgd1k-root "/path/to/mgd1k-official/Expore MGD1k Dataset" --mask-type gland --device cuda --batch-size 8 --epochs 50 --foreground-weight 5
```

Hard-negative learning run:

```bash
python plans/meibomianGlandImprovment/train.py --mgd1k-root "/path/to/mgd1k-official/Expore MGD1k Dataset" --mask-type gland --device cuda --batch-size 8 --epochs 50 --foreground-weight 5 --negative-weight 0.2 --hard-negative-percent 0.1 --hard-negative-min-prob 0.3
```

Suggested first sweep:

- `--negative-weight 0.1`
- `--negative-weight 0.2`
- `--negative-weight 0.5`

Keep the same split, foreground weight, and epoch count when comparing with baseline runs.

Structure-aware clDice run:

```bash
python plans/meibomianGlandImprovment/train.py --mgd1k-root "/path/to/mgd1k-official/Expore MGD1k Dataset" --mask-type gland --device cuda --batch-size 8 --epochs 50 --foreground-weight 5 --cldice-weight 0.1
```

Suggested first clDice sweep:

- `--cldice-weight 0.05`
- `--cldice-weight 0.1`
- `--cldice-weight 0.2`

Start without `--negative-weight` so the effect of the structure loss is isolated.
