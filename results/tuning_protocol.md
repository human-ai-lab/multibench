# Backbone tuning protocol (fixed 2026-10-10 20:47, before any tuning result was read)

Grid (seed 0, recipe = lung mask + strong aug + source-balanced sampling, 8 epochs x 150 steps, 4 LOCO folds qatar/nlm/tbx11k/pakistan):
- vit, raddino: backbone lr {2e-5, 5e-5, 1e-4} x unfrozen blocks {6, 12}
- resnet50, xrv_densenet: backbone lr {5e-5, 2e-4, 1e-3} x unfrozen stages {layer4 | layer3+4; denseblock4 | +denseblock3}
Selection per backbone: the config with the highest mean LOCO AUROC over the 4 public corpora. Configs within 0.01 of the best
are treated as tied and the tie is broken toward fewer unfrozen blocks, then lower lr. As a robustness check we also report the
config that an NLM+TBX11K-only criterion would pick (Pakistan and Qatar labels are artefact-predictable; see artefact_adjusted.md).
CIDRZ is never used for selection. The grid also re-runs the original config of each backbone (lr 5e-5, 6 blocks / layer4) with the
checkpoint-restore fix (see below), so pre- and post-fix numbers are never mixed in a table.

Bug fixed before this grid: `loco_train` restored only parameters whose names contained `vit.encoder` or `head` at the best epoch.
torchvision ViT was unaffected (all its trainable tensors match), but ResNet50 / xrv-DenseNet tails (and BN statistics) and RAD-DINO's
final `layernorm` stayed at their LAST-epoch values. All earlier ResNet50, xrv-DenseNet and RAD-DINO numbers (and the CIDRZ fusion
studies that use RAD-DINO experts) are therefore superseded by the re-runs in `results/loco_tune`, `results/loco_final`.
