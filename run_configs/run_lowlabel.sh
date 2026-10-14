#!/bin/bash
cd ~/scREFINE/scREFINE_for_review_experiments
mkdir -p result_low2pct logs_low2pct
for s in 17 42 109 172; do
  for d in Brain Fat Retina Breast_carcinoma Quadriceps_tendon; do
    nohup python3 run_scREFINE.py --data data/${d}.h5ad --output-dir result_low2pct/${d}_soft_s${s} --labeled-ratio 0.02 --seed ${s} --split-seed ${s} --epochs 100 --w-structure 0.1 > logs_low2pct/${d}_soft_s${s}.log 2>&1 &
    nohup python3 run_scREFINE.py --data data/${d}.h5ad --output-dir result_low2pct/${d}_hard_s${s} --labeled-ratio 0.02 --seed ${s} --split-seed ${s} --epochs 100 --w-structure 0.1 --hard-pl > logs_low2pct/${d}_hard_s${s}.log 2>&1 &
  done
done
echo "low-label (2%) soft-vs-hard: 40 run launched"
