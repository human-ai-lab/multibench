# usage: run_loco_grid.sh "<name>|<flags>" ...   (each spec run for 5 targets x 3 seeds)
COMMON="--epochs 8 --steps-per-epoch 150"
for spec in "$@"; do
  name="${spec%%|*}"; flags="${spec#*|}"
  for seed in ${SEEDS:-0 1}; do for t in ${TARGETS:-qatar nlm tbx11k pakistan cidrz}; do
    [ -f results/loco/${name}_${t}_s${seed}.json ] && continue
    .venv/bin/python -m datasets.tb_multi.loco_train --target $t --name $name --seed $seed $COMMON $flags 2>&1 | grep -E "RESULT|Error|Traceback" >> results/loco_grid.log
  done; done
done
echo "DONE $*" >> results/loco_grid.log
