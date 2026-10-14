#!/bin/bash
cd ~/benchmark_methods/scGPT_scAGCL
source .venv_scagcl/bin/activate
SP=~/scREFINE/scREFINE_for_review_experiments/splits
GI=~/scREFINE/scREFINE_for_review_experiments/scgpt_inputs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=""
mkdir -p logs result
for s in 17 42 109 172; do
  nohup python scAGCL.py --data_file ${GI}/Brain_s${s}.h5 --use_scgpt --num_cluster 22 --random_seed ${s} --splits_npz ${SP}/Brain_s${s}.npz > logs/Brain_s${s}.log 2>&1 &
  nohup python scAGCL.py --data_file ${GI}/Fat_s${s}.h5 --use_scgpt --num_cluster 34 --random_seed ${s} --splits_npz ${SP}/Fat_s${s}.npz > logs/Fat_s${s}.log 2>&1 &
  nohup python scAGCL.py --data_file ${GI}/Retina_s${s}.h5 --use_scgpt --num_cluster 71 --random_seed ${s} --splits_npz ${SP}/Retina_s${s}.npz > logs/Retina_s${s}.log 2>&1 &
  nohup python scAGCL.py --data_file ${GI}/Breast_carcinoma_s${s}.h5 --use_scgpt --num_cluster 19 --random_seed ${s} --splits_npz ${SP}/Breast_carcinoma_s${s}.npz > logs/Breast_s${s}.log 2>&1 &
  nohup python scAGCL.py --data_file ${GI}/Quadriceps_tendon_s${s}.h5 --use_scgpt --num_cluster 22 --random_seed ${s} --splits_npz ${SP}/Quadriceps_tendon_s${s}.npz > logs/Quad_s${s}.log 2>&1 &
done
echo "scGPT+scAGCL: 20 run launched"