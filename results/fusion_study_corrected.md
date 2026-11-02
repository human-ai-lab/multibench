
### image expert: raddino5f  (alone AUROC 0.825)

| arm | AUROC | UAR@0.5 or sign | ΔAUROC vs same-method w/o image |
|---|---|---|---|
| text | 0.782±0.010 | 0.700 |  |
| audio | 0.610±0.012 | 0.580 |  |
| image | 0.824±0.001 | 0.713 |  |
| stack|text+audio | 0.775±0.010 | 0.713 |  |
| relw|text+audio | 0.782±0.008 | 0.695 |  |
| concat|text+audio | 0.768±0.009 | 0.713 |  |
| stack|text+image | 0.845±0.005 | 0.771 |  |
| relw|text+image | 0.852±0.004 | 0.739 |  |
| concat|text+image | 0.842±0.007 | 0.771 |  |
| stack|text+audio+image | 0.840±0.006 | 0.768 | +0.061 [+0.016,+0.106] |
| relw|text+audio+image | 0.848±0.004 | 0.741 | +0.062 [+0.023,+0.100] |
| concat|text+audio+image | 0.816±0.010 | 0.753 | +0.046 [+0.019,+0.073] |
| mdmlp|text+audio | 0.744±0.013 | 0.673 |  |
| mdmlp|text+audio+image | 0.802±0.009 | 0.738 | +0.053 [+0.024,+0.082] |
| mdmlp|full, audio missing at test | 0.825±0.006 | 0.757 |  |
| mdmlp|full, text missing at test | 0.747±0.012 | 0.698 |  |
| mdmlp|full, image missing at test | 0.741±0.010 | 0.669 |  |

### image expert: ens  (alone AUROC 0.828)

| arm | AUROC | UAR@0.5 or sign | ΔAUROC vs same-method w/o image |
|---|---|---|---|
| text | 0.782±0.010 | 0.700 |  |
| audio | 0.610±0.012 | 0.580 |  |
| image | 0.827±0.001 | 0.733 |  |
| stack|text+audio | 0.775±0.010 | 0.713 |  |
| relw|text+audio | 0.782±0.008 | 0.695 |  |
| concat|text+audio | 0.768±0.009 | 0.713 |  |
| stack|text+image | 0.844±0.006 | 0.779 |  |
| relw|text+image | 0.851±0.004 | 0.732 |  |
| concat|text+image | 0.843±0.008 | 0.780 |  |
| stack|text+audio+image | 0.838±0.006 | 0.777 | +0.058 [+0.013,+0.103] |
| relw|text+audio+image | 0.847±0.004 | 0.733 | +0.060 [+0.021,+0.098] |
| concat|text+audio+image | 0.816±0.009 | 0.760 | +0.045 [+0.017,+0.073] |
| mdmlp|text+audio | 0.744±0.013 | 0.673 |  |
| mdmlp|text+audio+image | 0.800±0.010 | 0.741 | +0.051 [+0.022,+0.081] |
| mdmlp|full, audio missing at test | 0.825±0.007 | 0.763 |  |
| mdmlp|full, text missing at test | 0.752±0.013 | 0.702 |  |
| mdmlp|full, image missing at test | 0.737±0.012 | 0.668 |  |
