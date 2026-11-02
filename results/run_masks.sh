set -e
for s in 0 1 2; do
  .venv/bin/python -m datasets.tb_cxr_qatar.cross_dataset_eval --qatar-config configs/tb_cxr_qatar.yaml --cidrz-path data/google_health/tb_dataset.pkl --seed $s --save results/models/tb_cxr_qatar_whole_s$s.pt > results/whole_s$s.log 2>&1
  .venv/bin/python -m datasets.tb_cxr_qatar.cross_dataset_eval --qatar-config configs/tb_cxr_qatar_lungmask.yaml --cidrz-path data/google_health/tb_dataset_lungmask.pkl --seed $s --save results/models/tb_cxr_qatar_lungmask_s$s.pt > results/lungmask_s$s.log 2>&1
done
.venv/bin/python -m datasets.tb_cxr_qatar.lung_mask_eval --whole results/models/tb_cxr_qatar_whole_s{0,1,2}.pt --lungmask results/models/tb_cxr_qatar_lungmask_s{0,1,2}.pt > results/lung_mask_eval_replicated.txt 2>&1
echo CHAIN_DONE >> results/lung_mask_eval_replicated.txt
