# Candidate extra external corpus: Kaggle "Indian TB and Normal X rays" (2026-10-10) - REJECTED as a TB target

Source: kagglehub `mohdabdulrahman1/indian-tb-and-normal-x-rays` (train+val merged; 16,807 JPEG, all 512x512;
8,232 TB / 8,575 normal). **Provenance is not documented by the uploader** (the "Indian" label could not be verified; file
names mix hash-named `*_Normal/_Abnormal.jpg` and `Normal_Training_*/AB_training_*` patterns, i.e. an assembled/remixed set).
It is therefore used only as an additional *external* target (never trained on, never used for selection), and results on it
are reported as "india (unverified provenance)".

Cross-corpus duplicate check (same detector as `tb_multi_dedup_report.md`: 64-bit pHash Hamming<=8 + 64x64 correlation>=0.97):
0 matches against qatar, shenzhen, montgomery, tbx11k, pakistan, cidrz. xrv-DenseNet nearest-neighbour cosine (3,000 random india
images) max: qatar 0.964, tbx11k 0.961, montgomery 0.951, pakistan 0.950, shenzhen 0.934, cidrz 0.918 (copies of the same image are
>=0.997 in the calibration control). Within india: 91 near-duplicate pairs; embedding-cosine clustering gives 15,531 clusters at
cos>=0.985 (14,964 singletons, largest cluster 216) and 16,742 at >=0.995, so the effective sample size is only mildly below n.
Limitation: heavy crops/flips are not covered; patient identifiers are unavailable, so patient-level independence is not guaranteed.

Shortcut controls (LOCO from the four public sources; `shortcut_audit_india.log`): mask-area 0.510, lung silhouette 0.610,
outer border 0.438 AUROC. india is therefore about as artefact-clean as CIDRZ (0.612/0.611/0.462) and much cleaner than
qatar (0.82/0.84/0.78) or pakistan (0.48/0.80/0.76).

## Label-validity check (why india is NOT used for any headline claim)
- Positive-class file names are `*_Abnormal.jpg` (5,569), `AB_training_*` (2,363) and `Abnormal_Training_*` (300): the "tb" folder is the
  uploader's *abnormal* class, not documented as microbiologically or radiologically confirmed TB. Negatives: `*_Normal.jpg` (5,392) and
  `Normal_Training_*` (3,183).
- The two naming sources are separable from image content alone (5-fold probe on xrv features: hash-Normal vs Normal_Training AUROC 0.937,
  hash-Abnormal vs AB_training 0.878, vs Abnormal_Training 0.850), i.e. at least two acquisition/source pools with different class mixes.
- A within-india 5-fold probe reaches AUROC 0.846, but a frozen xrv probe trained on the four public corpora reaches only 0.552 on india, and
  the smoke-test fine-tuned models scored 0.58-0.65 on india versus 0.72-0.83 on CIDRZ in the same runs. A TB detector transfers to CIDRZ
  but not to this label, consistent with "abnormal" != "TB".
Conclusion: usable at most as an exploratory stress test (`loco_train --eval-ext`); not a documented external cohort. A documented
cohort with TB labels needs credentialed access (VinDr-CXR / PadChest / NIAID TB Portals), which this environment does not have.
