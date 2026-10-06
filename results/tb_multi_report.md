# Cross-corpus and multimodal TB detection: experiment report (2026-10-06)

Code: `datasets/tb_multi/` (standardize, dedup, embed, masks, loco_probe, shortcut_audit, loco_train, summarize_loco,
fusion_study). Raw outputs: `results/loco/`, `results/loco_summary.md`, `results/fusion_study.md`,
`results/loco_probe_xrv.log`, `results/shortcut_audit.log`, `results/tb_multi_dedup_report.md`.

## Corpora (after dedup; TBX11K = active TB vs healthy + sick-non-TB, latent/uncertain excluded)
qatar 4187 (16.4% TB) | nlm = montgomery 138 + shenzhen 661 (49%) | tbx11k 8029 (8.2%) | pakistan 2973 (82.9%, ~256px JPEG)
| cidrz 364 (16.5%, external target, never used for selection).

## 1. Shortcuts (xrv-DenseNet frozen features / simple controls)
- Corpus identity from embedding: 96.8% accuracy; corpus id alone predicts the TB label (pooled 5-fold AUROC 0.831).
- LOCO AUROC of controls (qatar / nlm / tbx11k / pakistan / cidrz): lung-silhouette-only 0.84/0.71/0.60/0.80/0.61;
  outer-border-pixels-only 0.78/0.53/0.61/0.76/0.46. Frozen xrv probe (raw) 0.84/0.73/0.69/0.68/0.70.
  => Qatar and Pakistan held-out scores are largely explainable by silhouette/border artefacts.
- Qatar shows no near-duplicates of Montgomery/Shenzhen (contradicts the earlier-session assumption).

## 2. Fine-tuned ViT-B/16, LOCO (AUROC, mean over 2 seeds; full table with CIs in loco_summary.md)
Best by LOCO mean (4 corpora): sbal_aug_lung_dstd 0.849, sbal_aug_lung 0.846, sbal_lung 0.835 vs source-balanced
baseline 0.782 and ERM 0.786. Lung masking and strong augmentation (incl. resolution jitter + border erasing) help
on 3/4 corpora; DG losses (GroupDRO, V-REx, CORAL, DANN) are not better than source-balanced sampling on average
and trade corpora against each other. Pakistan behaves differently (ERM 0.91, but masked/augmented 0.74-0.81):
it is the least trustworthy target (silhouette-only 0.80).

## 3. CIDRZ multimodal (image expert trained without CIDRZ; text/audio fit in CV; 10x5-fold)
image alone: ERM 0.760, sbal_aug_lung 0.809. text 0.782, audio 0.610, text+audio 0.775.
Best fusion: reliability-weighted text+image 0.840 (vs text 0.782); text+audio+image vs text+audio:
+0.048 [+0.003,+0.093] (stack), +0.050 [+0.009,+0.091] (relw), +0.041 [+0.013,+0.069] (modality-dropout MLP).
Audio adds nothing. With ERM image expert the gain is not significant (+0.026 [-0.015,+0.062]).

## Caveats
- ViT config (6 unfrozen blocks, lr 5e-5) came from the earlier CIDRZ-informed sweep; method choice here used LOCO only.
- 2 seeds per cell; CIDRZ has 60 positives (wide CIs). The UAR column in fusion_study.md thresholds raw scores at
  0 and is not meaningful for relw; use AUROC.
- Not done: TBX11K lesion-box supervision, MixStyle, CNN/foundation-model fine-tuning comparison, calibration analysis.
