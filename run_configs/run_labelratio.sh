#!/bin/bash
cd ~/scREFINE/scREFINE_for_review_experiments
mkdir -p result_ratio logs_ratio
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
for r in 0.1 0.2 0.5; do
  for s in 17 42 109 172; do
    for d in Brain Fat Retina Breast_carcinoma Quadriceps_tendon; do
      nohup python3 run_scREFINE.py --data data/${d}.h5ad --output-dir result_ratio/${d}_r${r}_s${s} --labeled-ratio ${r} --seed ${s} --split-seed ${s} --epochs 100 --w-structure 0.1 > logs_ratio/${d}_r${r}_s${s}.log 2>&1 &
    done
  done
done
echo "label-ratio: 60 run launched"