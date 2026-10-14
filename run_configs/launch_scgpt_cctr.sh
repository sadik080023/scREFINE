#!/bin/bash
cd ~/benchmark_methods/scGPT_scCCTR
source .venv_sccctr/bin/activate
SP=~/scREFINE/scREFINE_for_review_experiments/splits
GI=~/scREFINE/scREFINE_for_review_experiments/scgpt_inputs
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=""
mkdir -p logs out_v2
for s in 17 42 109 172; do
  for d in Brain Fat Retina Breast_carcinoma Quadriceps_tendon; do
    nohup bash -c "python run_scgpt_sccctr_direct.py --data_path ${GI}/${d}_s${s}.h5 --out_dir out_v2/${d}_s${s} --seed ${s} --splits_npz ${SP}/${d}_s${s}.npz && python run_scgpt_sccctr_transformer.py --data_path ${GI}/${d}_s${s}.h5 --pseudo_path out_v2/${d}_s${s}/scgpt_sccctr_direct_consensus_clusters.csv --out_dir out_v2/${d}_s${s}_tr --seed ${s} --splits_npz ${SP}/${d}_s${s}.npz" > logs/${d}_s${s}.log 2>&1 &
  done
done
echo "scGPT+scCCTR: 20 chain launched (direct -> transformer)"