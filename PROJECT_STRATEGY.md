# Project Strategy: Improve Results for the MGD-1k Dataset

## Objective

Focus on the dataset linked in `plans/blink_improvement/For Regiss - dataset exploitable`:
- MGD-1k: 1,000 infrared meibomian gland images with segmentation masks and graded meiboscore.

The goal is to define a reproducible research and modeling pipeline that improves performance on this dataset and creates a path for broader dry eye-related ocular imaging applications.

## 1. Clarify the task

### Primary task
- Meibomian gland analysis using infrared meibography.
- Target outcomes:
  - Accurate gland segmentation.
  - Robust meiboscore prediction / MGD severity classification.
  - Quantitative gland dropout measurement.

### Associated tasks
- Image preprocessing and artifact mitigation.
- Multi-task modeling: segmentation + classification/regression.
- Data quality assessment and augmentation.

## 2. Data audit

### Actions
- Download and inspect the MGD-1k dataset.
- Verify annotation consistency and mask alignment.
- Measure class balance and image quality.
- Identify failure modes: low contrast, partial views, noise, specular reflections.

### Key checks
- Number of samples by meiboscore class.
- Presence of multiple views or image cropping differences.
- Mask coverage and annotation quality.
- Any missing or corrupted files.

## 3. Build a strong baseline

### Recommended baseline models
- Segmentation: U-Net with ResNet34 encoder, U-Net++ with EfficientNet-B0 encoder.
- Classification/regression head: global pooling + MLP on encoder features for meiboscore prediction.
- Multi-task setup: single encoder, segmentation decoder, and separate score head.

### Preprocessing pipeline
- Standardize all images to 320x320 with padding or center crop to preserve gland detail.
- Convert the dataset to a single normalized grayscale channel.
- Apply CLAHE per image, then normalize by dataset mean and std.
- Keep preprocessing deterministic for evaluation.

### Baseline training recipe
- Use 5-fold stratified cross-validation on meiboscore labels.
- Train 60 epochs with early stopping based on validation Dice.
- Loss for segmentation: Dice + BCE.
- Loss for score: MSE for regression or cross-entropy for 4-class classification.
- Optimizer: AdamW with lr 1e-4, weight decay 1e-5, cosine annealing.
- Batch size: 8–16 depending on GPU memory.

## 4. Improve through targeted techniques

### Option A — data-centric gains
1. Precise augmentation recipe
   - Spatial: rotate ±12°, translate ±8%, scale 0.95–1.05.
   - Intensity: brightness ±15%, contrast ±18%, gamma 0.9–1.1.
   - Distortions: elastic transform with alpha 20–40, sigma 6–8.
   - Occlusion: random cutout or erasing of a 10–20% patch.
   - Infrared domain: random local contrast stretch and low-light intensity variation.
2. Data reuse and synthetic diversity
   - Use mixup/CutMix on image-mask pairs preserving segmentation labels.
   - If polygon masks are available, perform small affine transformations on masks and reapply.
3. Training improvements without dataset modification
   - Apply Test-Time Augmentation (TTA) on validation/inference.
   - Use class-weighting or focal loss for score imbalance.
   - Implement a dataset sampler that oversamples rare meiboscore classes.
4. Transfer learning and self-supervision
   - Initialize encoder from ImageNet or ocular imaging pretrained weights.
   - Pretrain on unlabeled infrared images from other sources with rotation prediction or contrastive objectives.
5. Validation and metrics
   - Track mask Dice, mask IoU, score MAE, and score F1 per fold.
   - Choose checkpoints by the combined score: 0.6 * Dice + 0.4 * normalized score metric.

### Option B — architecture-centric gains
1. Encoder upgrades
   - Replace ResNet34 with EfficientNet-B3, ConvNeXt-Tiny, or ResNeXt50.
   - If memory allows, try a Swin Transformer-style encoder.
2. Multi-task architecture design
   - Use a shared backbone with two heads: one decoder for masks, one dense head for meiboscore.
   - Add feature pyramid fusion between encoder stages and score head.
3. Decoder and loss enhancements
   - Use attention gates, squeeze-and-excitation blocks, or ASPP in the decoder.
   - Add boundary or edge loss: Dice + BCE + boundary IoU.
   - For ordinal score labels, use an ordinal regression loss.
4. Ensemble and model combination
   - Train at least two architectures with different inductive biases.
   - Ensemble predictions by averaging mask outputs and score logits.

### Explicit decision rule
- Start with Option A and run 3 focused experiments: baseline, stronger augmentation, and transfer learning.
- If validation Dice improves by <1.5% over the baseline after these experiments, move to Option B.
- Keep all experiments using the same split and evaluation script.

## 5. Expand dataset context

### Transfer learning
- Pretrain on general ocular imaging or medical imaging datasets as available.
- Prefer pretrained weights from other infrared or eye-image tasks over generic natural images when possible.

### Data fusion
- Add external ocular datasets only as source of domain adaptation; do not change the original MGD-1k labels.
- Use external data for encoder pretraining or self-supervised representation learning.

### External validation
- Reserve a small set of any compatible external meibography images for final held-out testing.
- If no external dataset exists, use the least-similar MGD-1k fold as an approximation.

### Transfer learning
- Pretrain on general ocular imaging or medical imaging datasets.
- If possible, use self-supervised pretraining on unlabeled infrared ocular images.

### Data fusion
- Explore combining MGD-1k with other eye datasets in the workspace, especially if related to meibography or tear film imaging.
- Use dry eye datasets such as tear meniscus segmentation for general ocular domain adaptation.

### External validation
- Search for compatible external samples or small public meibomian gland datasets.
- Use external test images to assess generalization beyond MGD-1k.

## 6. Evaluation metrics

### Segmentation
- Dice coefficient.
- Jaccard index.
- Hausdorff distance or boundary IoU if available.

### Classification / regression
- Accuracy, F1 score, macro-averaged precision/recall.
- AUC for binary or ordinal thresholds.
- Mean absolute error (MAE) or mean squared error (MSE) for score regression.

### Deployment readiness
- Model inference speed and memory footprint.
- Robustness to image noise and acquisition variation.
- Explainability: visualize segmentation outputs and attention if using transformer-based models.

## 7. Workflow and documentation

### Reproducible workflow
- Store data processing scripts in `scripts/`.
- Keep dataset research and results notes in `plans/blink_improvement/`.
- Maintain summaries and methodology in `docs/`.
- Use `reports/weekly/` for progress updates and results presentation.

### Deliverables
- `scripts/` pipeline for dataset loading, preprocessing, training, and evaluation.
- `docs/` documentation of dataset choices, model design, and result comparisons.
- `reports/weekly/` status reports showing milestones, next steps, and metric tracking.

## 8. Immediate next actions

1. Open `plans/blink_improvement/For Regiss - dataset exploitable` and confirm the exact MGD-1k dataset structure.
2. Create a data inventory script in `scripts/` for the dataset contents.
3. Prototype a baseline segmentation model using U-Net or nnU-Net.
4. Track improvements using a consistent validation protocol.

## Notes

- The current workspace now has a dedicated plan location: `plans/blink_improvement/`.
- The dataset improvement strategy should be executed in a reproducible, documented pipeline so progress can be measured clearly.
