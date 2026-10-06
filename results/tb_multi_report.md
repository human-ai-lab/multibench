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

## 4. Backbones and lesion supervision (same recipe: lung mask + strong aug + source-balanced; 2 seeds; `loco_summary_all.md`)
LOCO mean (qatar/nlm/tbx11k/pakistan): RAD-DINO (CXR foundation model, 224px, last 6 blocks) 0.906 > ViT-B/16 ImageNet 0.846
> ResNet50 0.786 ~ xrv-DenseNet121 0.784. RAD-DINO beats the ViT on all four LOCO corpora (+0.03 to +0.08, CIs exclude 0);
on CIDRZ all backbones sit at 0.74-0.82 with no significant difference (xrv-DenseNet 0.817, RAD-DINO 0.798, ViT 0.805).
Lesion-box supervision (TBX11K active-TB boxes -> patch-token auxiliary loss): no gain for the CLS score (ViT: qatar -0.007,
nlm +0.001, pakistan +0.050, cidrz -0.012; RAD-DINO: all within +-0.01 except pakistan -0.014, nlm -0.016). The patch-max
lesion score is sometimes higher (e.g. ViT cidrz 0.831 vs 0.792, RAD-DINO pakistan 0.873 vs 0.782) but worse elsewhere
(ViT pakistan 0.653) and has no CIs/selection protocol: exploratory only. The tbx11k fold cannot use the box labels
(held out), so it reproduces the baseline.
CIDRZ fusion with the LOCO-selected RAD-DINO expert (fusion_study_raddino.md): relw text+image 0.842, text+audio+image vs
text+audio +0.052 [+0.017,+0.087]; same picture as the ViT expert -- CIDRZ looks capped near 0.80-0.84 AUROC.

## 5. MixStyle, calibration, final CIDRZ expert (`loco_summary_all.md`, `calibration.md`, `fusion_study_final.md`)
MixStyle (p=0.5, alpha=0.1, on activations entering the trainable tail; 2 seeds): ViT 0.874 vs 0.846 LOCO mean without it
(Pakistan +0.10, Qatar +0.011, NLM/TBX11K ~0, CIDRZ ~0); RAD-DINO 0.909 vs 0.906 (no real change: Pakistan +0.024, CIDRZ -0.026
[-0.045,-0.008]); ResNet50 0.768 vs 0.786 (hurts). So MixStyle helps the weaker ImageNet ViT but not the CXR foundation model,
and the ViT gain is mostly on Pakistan, the corpus with the strongest artefacts.
Calibration (raw logit margins; training used source-balanced cells): every expert is over-confident on held-out corpora
(logistic calibration slope 0.2-0.6; ECE 0.04-0.25, up to 0.49-0.59 on CIDRZ for RAD-DINO/ERM). Correcting the prior shift with
the target's true prevalence (oracle) only removes part of it. On CIDRZ the raw-probability Brier (0.22-0.45) is worse than
predicting the prevalence (0.138); the discriminative ranking transfers (AUROC 0.8), probabilities do not. Threshold transfer
(90%-sensitivity threshold from held-in validation) gives target sensitivity 0.6-0.9 and specificity 0.55-0.98 depending on
corpus, e.g. RAD-DINO on CIDRZ 0.73/0.77. => deployment needs per-site recalibration (a few labelled local cases); the
fusion stacker already supplies it for CIDRZ.
Final CIDRZ expert (trained on all four source corpora pooled, never CIDRZ): RAD-DINO 5 seeds AUROC 0.813, RAD-DINO+ViT
ensemble (5 seeds each) 0.825. Fusion (fusion_study_final.md): relw text+image 0.845-0.848 (text 0.782); text+audio+image vs
text+audio +0.055 [+0.011,+0.097] (stack), +0.057 [+0.019,+0.094] (relw), +0.048 [+0.020,+0.077] (MD-MLP).
Note: the cidrz-target runs already pooled all four source corpora; "pooled final expert" here means more seeds + ensembling.

## Caveats
- ViT config (6 unfrozen blocks, lr 5e-5) came from the earlier CIDRZ-informed sweep; method choice here used LOCO only.
- 2 seeds per cell; CIDRZ has 60 positives (wide CIs). The UAR column in fusion_study.md thresholds raw scores at
  0 and is not meaningful for relw; use AUROC.
- RAD-DINO is run at 224px although it was trained at 518px; backbones share one lr (5e-5) and unfreezing rule per family, untuned.
- Calibration is evaluated on image-expert logits only (fused-probability calibration not analysed); MixStyle/ensemble have 2/5 seeds.
