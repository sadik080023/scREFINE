#!/bin/bash
cd ~/scREFINE/scREFINE_for_review_experiments
for s in 17 42 109 172; do
  nohup python3 k_estimation.py --data data/Brain.h5ad --emb result/Brain_s${s}/fused_embeddings.npy --split-seed ${s} --true-k 22 > logs/ke_Brain_s${s}.log 2>&1 &
  nohup python3 k_estimation.py --data data/Fat.h5ad --emb result/Fat_s${s}/fused_embeddings.npy --split-seed ${s} --true-k 34 > logs/ke_Fat_s${s}.log 2>&1 &
  nohup python3 k_estimation.py --data data/Retina.h5ad --emb result/Retina_s${s}/fused_embeddings.npy --split-seed ${s} --true-k 72 > logs/ke_Retina_s${s}.log 2>&1 &
  nohup python3 k_estimation.py --data data/Breast_carcinoma.h5ad --emb result/Breast_carcinoma_s${s}/fused_embeddings.npy --split-seed ${s} --true-k 19 > logs/ke_Breast_s${s}.log 2>&1 &
  nohup python3 k_estimation.py --data data/Quadriceps_tendon.h5ad --emb result/Quadriceps_tendon_s${s}/fused_embeddings.npy --split-seed ${s} --true-k 22 > logs/ke_Quad_s${s}.log 2>&1 &
done
echo "K-estimation: 20 ta launch hoye geche"