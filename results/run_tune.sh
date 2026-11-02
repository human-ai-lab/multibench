#!/bin/bash
# usage: run_tune.sh <worker> <n_workers>   -- backbone tuning grid, selected on the 4 source-corpus LOCO folds only
COMMON="--epochs 8 --steps-per-epoch 150 --sampler sbal --aug strong --input lung --outdir results/loco_tune"
mkdir -p results/loco_tune
CONFIGS=()
for lr in 2e-5 5e-5 1e-4; do for u in 6 12; do
  CONFIGS+=("tune_vit_lr${lr}_u${u}|--backbone vit --backbone-lr $lr --unfreeze-blocks $u")
  CONFIGS+=("tune_raddino_lr${lr}_u${u}|--backbone raddino --backbone-lr $lr --unfreeze-blocks $u")
done; done
for lr in 5e-5 2e-4 1e-3; do for u in 1 2; do
  CONFIGS+=("tune_resnet50_lr${lr}_c${u}|--backbone resnet50 --backbone-lr $lr --cnn-unfreeze $u")
  CONFIGS+=("tune_xrv_lr${lr}_c${u}|--backbone xrv_densenet --backbone-lr $lr --cnn-unfreeze $u")
done; done
w=$1; n=$2
for i in "${!CONFIGS[@]}"; do
  [ $((i % n)) -eq $w ] || continue
  spec="${CONFIGS[$i]}"; name="${spec%%|*}"; flags="${spec#*|}"
  for t in qatar nlm tbx11k pakistan; do
    [ -f results/loco_tune/${name}_${t}_s0.json ] && continue
    .venv/bin/python -m datasets.tb_multi.loco_train --target $t --name $name --seed 0 $COMMON $flags 2>&1 | grep -E "RESULT|Error|Traceback" >> results/loco_tune.log
  done
done
echo "DONE worker $w" >> results/loco_tune.log
