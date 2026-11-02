#!/bin/bash
# usage: run_vindr.sh <worker> <n_workers>
# Same 8 tuned specs / 5 seeds as run_final.sh, but with --target vindr: trained on the four sources only (identical held-in split and
# training set as the cidrz runs), scored on VinDr-CXR; --eval-ext also re-scores cidrz (sanity check vs results/loco_final) and india.
# Seed-major order so a partial run still has complete seeds. Checkpoints go to /data so another cohort can be scored without retraining.
OUT=results/loco_vindr; mkdir -p $OUT
COMMON="--epochs 8 --steps-per-epoch 150 --outdir $OUT --eval-ext --ckpt-dir /data/tb_multi_ckpt"
BB="--sampler sbal --aug strong --input lung"
SPECS=(
 "f_raddino|--backbone raddino --backbone-lr 1e-4 $BB"
 "f_ms_raddino|--backbone raddino --backbone-lr 1e-4 --mixstyle 0.5 $BB"
 "f_vit|--backbone vit --backbone-lr 1e-4 $BB"
 "f_resnet50|--backbone resnet50 --backbone-lr 1e-3 --cnn-unfreeze 2 $BB"
 "f_xrv|--backbone xrv_densenet --backbone-lr 1e-3 --cnn-unfreeze 1 $BB"
 "f_erm|--backbone vit --backbone-lr 1e-4 --sampler random --aug none --input whole"
 "f_sbal|--backbone vit --backbone-lr 1e-4 --sampler sbal --aug none --input whole"
 "f_ms_vit|--backbone vit --backbone-lr 1e-4 --mixstyle 0.5 $BB"
)
JOBS=()
for seed in 0 1 2 3 4; do for spec in "${SPECS[@]}"; do JOBS+=("$spec|$seed"); done; done
w=$1; n=$2
for i in "${!JOBS[@]}"; do
  [ $((i % n)) -eq $w ] || continue
  IFS='|' read name flags seed <<< "${JOBS[$i]}"
  [ -f $OUT/${name}_vindr_s${seed}.json ] && continue
  .venv/bin/python -m datasets.tb_multi.loco_train --target vindr --name $name --seed $seed $COMMON $flags 2>&1 | grep -E "RESULT|EXT|Error|Traceback|error" >> results/loco_vindr.log
done
echo "DONE worker $w" >> results/loco_vindr.log
