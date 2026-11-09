#!/bin/bash
# usage: run_vindr448.sh <worker> <n_workers> -- RAD-DINO at 448 px (TB_RES=448 cache), tuned config as run_448.sh, target vindr, seeds 0-4.
# --eval-ext also re-scores cidrz at 448 (sanity check vs results/loco_448/r448_cidrz_*; same four-source training set).
OUT=results/loco_vindr448; mkdir -p $OUT
export TB_RES=448
w=$1; n=$2; i=0
for seed in 0 1 2 3 4; do
  if [ $((i % n)) -eq $w ] && [ ! -f $OUT/r448_vindr_s${seed}.json ]; then
    .venv/bin/python -m datasets.tb_multi.loco_train --target vindr --name r448 --seed $seed --epochs 8 --steps-per-epoch 150 --outdir $OUT \
      --backbone raddino --backbone-lr 1e-4 --sampler sbal --aug strong --input lung --img-size 448 --batch-size 32 --eval-ext 2>&1 | grep -E "RESULT|EXT|Error|Traceback" >> results/loco_vindr448.log
  fi
  i=$((i+1))
done
echo "DONE worker $w" >> results/loco_vindr448.log
