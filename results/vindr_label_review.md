## VinDr-CXR label audit (f_raddino, seed-averaged score)

| positive tier | n | AUROC vs TB-free images of the same split | median score pct |
|---|---|---|---|
| train, 2 of 3 reads | 235 | 0.898 | 82 |
| train, 3 of 3 reads | 247 | 0.946 | 88 |
| test, consensus | 164 | 0.868 | 78 |

Unanimous minus 2-of-3 AUROC on the training split: +0.048 [+0.024, +0.073] (bootstrap, shared negatives).

positives, low-scoring: mean other-finding rate = Lung cavity 0.04, Infiltration 0.17, Consolidation 0.00, Nodule Mass 0.04, Pulmonary fibrosis 0.67, Calcification 0.17, Pleural effusion 0.12, Pleural thickening 0.33, No finding 0.00

negatives, high-scoring: mean other-finding rate = Lung cavity 0.00, Infiltration 0.08, Consolidation 0.04, Nodule Mass 0.33, Pulmonary fibrosis 0.17, Calcification 0.04, Pleural effusion 0.21, Pleural thickening 0.33, No finding 0.00
