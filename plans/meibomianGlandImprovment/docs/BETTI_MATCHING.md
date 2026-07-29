# Betti matching H0

Training uses the official C++ implementation from
`https://github.com/nstucki/Betti-Matching-3D`, tested against commit
`db623be7456ae451beb422af3bbcea94e9863a8f`. The optional loss is restricted
to homology dimension zero (connected components). It is evaluated on the full
eyelid ROI rather than patches.

The default `--betti-weight 0.0` keeps the historical CE + Dice baseline
unchanged and does not require the external module. A positive weight requires
building the module and adding its build directory to `PYTHONPATH`.

The optional schedule arguments `--betti-warmup-epochs` and
`--betti-ramp-epochs` default to zero, preserving the historical constant-weight
behavior. For example, a warmup of 10 and a ramp of 10 keep Betti disabled for
epochs 1--10, then linearly increase its effective weight from epoch 11 until
the target weight is reached at epoch 20. The effective value is saved in the
history JSON and TensorBoard logs.

Topology fine-tuning can be initialized from an existing segmentation model
with `--initial-checkpoint`. Only the model weights are restored; the optimizer,
learning-rate scheduler and early-stopping state start fresh. Omitting the
option preserves the historical training-from-scratch behavior.

The C++ implementation computes persistence information on detached CPU
arrays. PyTorch then evaluates the H0 loss at the returned critical coordinates,
which preserves gradients with respect to the foreground probability map.

The validation and test payloads also report hard-mask metrics using
8-connectivity:

- `betti0_abs_error`
- `betti0_additional_components`
- `betti0_missing_components`

The additional/missing counts are signed count differences, not spatial Betti
matching errors. A fusion and an unrelated extra component can cancel in these
simple metrics, so qualitative review remains necessary.
