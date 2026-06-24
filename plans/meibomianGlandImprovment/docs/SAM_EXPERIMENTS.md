# SAM Experiments

This is a separate benchmark path for Segment Anything (SAM). It does not replace the DeepLabv3+ pipeline.

The first experiment uses an oracle box prompt built from the ground-truth mask. This is not a deployable automatic setting, but it answers a useful question: can SAM recover meibomian glands or eyelid masks when it receives a good prompt?

## Install

```bash
pip install -r plans/meibomianGlandImprovment/requirements-sam.txt
```

## Checkpoint

Create a checkpoint folder:

```bash
mkdir -p plans/meibomianGlandImprovment/models/sam
```

Recommended first checkpoint, small enough for quick testing:

```bash
wget -O plans/meibomianGlandImprovment/models/sam/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

Larger options:

```bash
wget -O plans/meibomianGlandImprovment/models/sam/sam_vit_l_0b3195.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
wget -O plans/meibomianGlandImprovment/models/sam/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Gland Oracle-Box Benchmark

```bash
CUDA_VISIBLE_DEVICES=1 python plans/meibomianGlandImprovment/evaluate_sam.py --mgd1k-root "/home/paul/Documents/MeibomianGland/meibomian-gland-model-improvement/data/Dataset/mgd1k-official/Expore MGD1k Dataset" --checkpoint plans/meibomianGlandImprovment/models/sam/sam_vit_b_01ec64.pth --model-type vit_b --mask-type gland --split test --device cuda --use-clahe
```

## Eyelid Oracle-Box Benchmark

```bash
CUDA_VISIBLE_DEVICES=1 python plans/meibomianGlandImprovment/evaluate_sam.py --mgd1k-root "/home/paul/Documents/MeibomianGland/meibomian-gland-model-improvement/data/Dataset/mgd1k-official/Expore MGD1k Dataset" --checkpoint plans/meibomianGlandImprovment/models/sam/sam_vit_b_01ec64.pth --model-type vit_b --mask-type eyelid --split test --device cuda --use-clahe
```

## Outputs

- Summary JSON: `plans/meibomianGlandImprovment/results/metrics/sam_summary_*`
- Per-image CSV: `plans/meibomianGlandImprovment/results/metrics/sam_per_image_*`
- Overlays: `plans/meibomianGlandImprovment/results/figures/sam_*`

Overlay colors:

- Green: ground truth
- Red: SAM prediction
- Yellow: overlap
- Blue box: oracle prompt
