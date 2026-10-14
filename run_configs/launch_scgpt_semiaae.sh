#!/bin/bash
cd ~/benchmark_methods/scGPT_scSemiAAE
source .venv_scsemiaae/bin/activate
SP=~/scREFINE/scREFINE_for_review_experiments/splits
GI=~/scREFINE/scREFINE_for_review_experiments/scgpt_inputs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=""
for s in 17 42 109 172; do
  for d in Brain Fat Retina Breast_carcinoma Quadriceps_tendon; do
    nohup python scSemiAAE.py --data_path ${GI}/${d}_s${s}.h5 --use_scgpt \
      --splits_npz ${SP}/${d}_s${s}.npz --seed ${s} \
      --label_cells 0.05 --batch_size 128 --fit_epochs 100 \
      > logs/${d}_s${s}.log 2>&1 &
  done
done
echo "scGPT+scSemiAAE: 20 run launched"