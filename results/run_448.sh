#!/bin/bash
# usage: run_448.sh <worker> <n_workers> -- RAD-DINO at native 448 px (TB_RES=448 cache), tuned config, seeds 0-4
OUT=results/loco_448; mkdir -p $OUT
export TB_RES=448
JOBS=(); for seed in 0 1 2 3 4; do for t in qatar nlm tbx11k pakistan cidrz; do JOBS+=("$seed|$t"); done; done
w=$1; n=$2
for i in "${!JOBS[@]}"; do
  [ $((i % n)) -eq $w ] || continue
  IFS='|' read seed t <<< "${JOBS[$i]}"
  [ -f $OUT/r448_${t}_s${seed}.json ] && continue
  .venv/bin/python -m datasets.tb_multi.loco_train --target $t --name r448 --seed $seed --epochs 8 --steps-per-epoch 150 --outdir $OUT \
    --backbone raddino --backbone-lr 1e-4 --sampler sbal --aug strong --input lung --img-size 448 --batch-size 32 2>&1 | grep -E "RESULT|Error|Traceback" >> results/loco_448.log
done
echo "DONE worker $w" >> results/loco_448.log
