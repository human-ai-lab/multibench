# Cross-corpus CXR-TB audit protocol

Each item states the check, how it is done here, and the measured result that motivates it
(files in `results/`; 5 seeds unless noted).

1. **Deduplicate across corpora before any split.** pHash + 64x64 correlation + embedding cosine
   (`datasets/tb_multi/dedup*`), with a positive control (planted duplicates must be recovered).
   *Why:* the same radiographs circulate in several public sets; leakage inflates external AUROC.
2. **Evaluate leave-one-corpus-out, not random splits.** `loco_train.py`, targets qatar/nlm/tbx11k/pakistan; CIDRZ is an external target.
   *Why:* ranking changes by corpus (ViT ERM: 0.87 on Pakistan, 0.69 on TBX11K; `loco_final_summary.md`).
3. **Run source-prediction and artefact-only controls.** Silhouette / border / area predictors under LOCO (`shortcut_audit.py`).
   *Why:* artefact-only AUROC is 0.85-0.97 (Qatar), 0.64-0.86 (NLM), 0.65-0.90 (TBX11K) in `artefact_adjusted.md`.
4. **Report artefact-stratified AUROC next to raw AUROC.** `artefact_adjust.py`.
   *Why:* RAD-DINO raw/adjusted: Qatar 0.974/0.935, NLM 0.936/0.898, CIDRZ 0.825/0.816 - the signal survives, by a margin you can see.
5. **Declare a corpus non-identifiable when artefacts predict the label.** Pakistan: artefact-only AUROC 1.000, 1% of case/control pairs usable.
   Do not report an adjusted number; do not use it as evidence of generalisation.
   *Why:* its 224->448 px gain (0.815->0.932, `loco_448.log`) cannot be attributed to pathology.
6. **Validate labels of every added cohort.** The India cohort was labelled "Abnormal", not TB, and was rejected (`tb_multi_india_report.md`).
7. **Fix the tuning protocol before looking at the external target.** Selection on the four public folds only, rule in `tuning_protocol.md`; report the alternative picks.
   *Why:* tuning gains were <=0.034, so untuned vs tuned comparisons are within noise on one seed.
8. **Use >=5 seeds and paired bootstrap CIs; claim only what excludes zero.** `summarize_loco.py`.
   *Why:* on CIDRZ all backbones are 0.77-0.81 with overlapping CIs; the RAD-DINO advantage (+0.02 to +0.07) holds on public folds only.
9. **Verify checkpoint restore and train/eval parity.** Full `state_dict` clone; check that restored weights reproduce the selected-epoch validation score.
   *Why:* a partial restore (only keys matching "vit.encoder"/"head") silently moved ResNet50 0.786->0.812 and xrv 0.784->0.779.
10. **Check resolution only after the artefact audit.** RAD-DINO 448 px: +0.03-0.04 on NLM/TBX11K, +0.004 on CIDRZ (2 seeds).
    Gains on public folds are not evidence of better transfer.
11. **Report calibration separately from discrimination.** ECE, Brier, slope. RAD-DINO on CIDRZ: raw ECE 0.50, 0.07 after Platt on 50-200 local labels.
12. **Report against the clinical target, with an oracle threshold.** WHO triage minimum: sensitivity >=90%, specificity >=70%.
    CIDRZ: best RAD-DINO gets 0.82 sens at 70% spec or 0.50 spec at 90% sens; local recalibration does not change this (P(meets TPP)=0.00).
    A model can be well calibrated and still miss the profile.
13. **Say which numbers were superseded.** Keep pre-fix and post-fix runs in separate directories and never in one table.
