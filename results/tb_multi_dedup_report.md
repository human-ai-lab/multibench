# Cross-corpus duplicate check (standardized 224x224, 2026-10-06)

Corpora: qatar 4200, montgomery 138, shenzhen 662, tbx11k 8400 (labelled TB11K images; test/extra skipped),
pakistan (Mendeley v3) 3008, cidrz 364.

pHash(Hamming<=8) + 64x64 correlation>=0.97 found 293 pairs, all within-corpus except 12 Qatar x TBX11K;
0 label conflicts. Drop list (later-listed corpus loses the copy): tbx11k 231, pakistan 35, qatar 13, shenzhen 1
(`results/tb_multi_dedup_drop.csv`).

Detector calibration: TBX11K's own `extra/mc+shenzhen` (400 known NLM copies) is recovered ~100% (329 Shenzhen +
73 Montgomery pairs); xrv-DenseNet cosine for these copies has p5 = 0.997.

Qatar vs NLM (Montgomery, Shenzhen): NO near-duplicates found. Pixel-correlation best match max 0.948 (copies
would be ~1.0); xrv-embedding cosine max 0.978 (Shenzhen->Qatar) vs 0.973 for the null corpus
(Pakistan->Qatar). Side-by-side of the top 12 matches (`results/images/dedup/`) shows different patients.
This contradicts the earlier-session assumption that Qatar's TB class pools Montgomery+Shenzhen.
Limitation: heavy crops/flips/mirroring are not covered.
