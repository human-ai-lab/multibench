#!/bin/bash
# usage: run_final.sh <worker> <n_workers>   -- 5-seed headline runs with the tuned backbone configs (results/tuning_summary.md, mean4 rule)
OUT=results/loco_final; mkdir -p $OUT
COMMON="--epochs 8 --steps-per-epoch 150 --outdir $OUT"
BB="--sampler sbal --aug strong --input lung"
SPECS=(
 "f_vit|--backbone vit --backbone-lr 1e-4 $BB"
 "f_raddino|--backbone raddino --backbone-lr 1e-4 $BB"
 "f_resnet50|--backbone resnet50 --backbone-lr 1e-3 --cnn-unfreeze 2 $BB"
 "f_xrv|--backbone xrv_densenet --backbone-lr 1e-3 --cnn-unfreeze 1 $BB"
 "f_erm|--backbone vit --backbone-lr 1e-4 --sampler random --aug none --input whole"
 "f_sbal|--backbone vit --backbone-lr 1e-4 --sampler sbal --aug none --input whole"
 "f_ms_vit|--backbone vit --backbone-lr 1e-4 --mixstyle 0.5 $BB"
 "f_ms_raddino|--backbone raddino --backbone-lr 1e-4 --mixstyle 0.5 $BB"
)
JOBS=()
for spec in "${SPECS[@]}"; do for seed in 0 1 2 3 4; do for t in qatar nlm tbx11k pakistan cidrz; do JOBS+=("$spec|$seed|$t"); done; done; done
w=$1; n=$2
for i in "${!JOBS[@]}"; do
  [ $((i % n)) -eq $w ] || continue
  IFS='|' read name flags seed t <<< "${JOBS[$i]}"
  [ -f $OUT/${name}_${t}_s${seed}.json ] && continue
  .venv/bin/python -m datasets.tb_multi.loco_train --target $t --name $name --seed $seed $COMMON $flags 2>&1 | grep -E "RESULT|Error|Traceback" >> results/loco_final.log
done
echo "DONE worker $w" >> results/loco_final.log
