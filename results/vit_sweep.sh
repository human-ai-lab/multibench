for cfg in "0 1e-5 _u0" "2 5e-5 _u2_lr5e-5" "6 1e-5 _u6" "6 5e-5 _u6_lr5e-5" "12 1e-5 _u12"; do
  set -- $cfg
  .venv/bin/python -m datasets.tb_cxr_qatar.vit_eval --seeds 0 1 2 --unfreeze-blocks $1 --backbone-lr $2 --tag $3 > results/vit_sweep$3.log 2>&1
done
echo SWEEP_DONE > results/vit_sweep_done.txt
