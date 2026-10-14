#!/bin/bash
cd ~/scREFINE/scREFINE_for_review_experiments
mkdir -p result_wsens logs_wsens
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
for w in 0.05 0.2 0.3; do
  for s in 17 42 109 172; do
    for d in Brain Fat Retina Breast_carcinoma Quadriceps_tendon; do
      nohup python3 run_scREFINE.py --data data/${d}.h5ad --output-dir result_wsens/${d}_w${w}_s${s} --labeled-ratio 0.05 --seed ${s} --split-seed ${s} --epochs 100 --w-structure ${w} > logs_wsens/${d}_w${w}_s${s}.log 2>&1 &
    done
  done
done
echo "w-sensitivity: 60 run launched"