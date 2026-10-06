
### image expert: bb_raddino  (alone AUROC 0.807)

| arm | AUROC | UAR@0.5 or sign | ΔAUROC vs same-method w/o image |
|---|---|---|---|
| text | 0.782±0.010 | 0.700 |  |
| audio | 0.610±0.012 | 0.580 |  |
| image | 0.806±0.001 | 0.701 |  |
| stack|text+audio | 0.775±0.010 | 0.713 |  |
| relw|text+audio | 0.782±0.008 | 0.695 |  |
| concat|text+audio | 0.768±0.009 | 0.713 |  |
| stack|text+image | 0.834±0.005 | 0.756 |  |
| relw|text+image | 0.842±0.004 | 0.733 |  |
| concat|text+image | 0.828±0.007 | 0.764 |  |
| stack|text+audio+image | 0.829±0.005 | 0.755 | +0.051 [+0.010,+0.088] |
| relw|text+audio+image | 0.838±0.004 | 0.725 | +0.052 [+0.017,+0.087] |
| concat|text+audio+image | 0.802±0.010 | 0.739 | +0.033 [+0.011,+0.056] |
| mdmlp|text+audio | 0.744±0.013 | 0.673 |  |
| mdmlp|text+audio+image | 0.791±0.010 | 0.729 | +0.042 [+0.016,+0.068] |
| mdmlp|full, audio missing at test | 0.812±0.006 | 0.750 |  |
| mdmlp|full, text missing at test | 0.730±0.012 | 0.690 |  |
| mdmlp|full, image missing at test | 0.743±0.010 | 0.674 |  |
