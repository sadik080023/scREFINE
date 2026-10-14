#!/bin/bash
cd ~/scREFINE/scREFINE_for_review_experiments
for s in 17 42 109 172; do
  nohup python3 raw_scgpt_kmeans.py --data data/Brain.h5ad --seed ${s} --n-classes 22 > logs/raw_Brain_s${s}.log 2>&1 &
  nohup python3 raw_scgpt_kmeans.py --data data/Fat.h5ad --seed ${s} --n-classes 34 > logs/raw_Fat_s${s}.log 2>&1 &
  nohup python3 raw_scgpt_kmeans.py --data data/Retina.h5ad --seed ${s} --n-classes 72 > logs/raw_Retina_s${s}.log 2>&1 &
  nohup python3 raw_scgpt_kmeans.py --data data/Breast_carcinoma.h5ad --seed ${s} --n-classes 19 > logs/raw_Breast_s${s}.log 2>&1 &
  nohup python3 raw_scgpt_kmeans.py --data data/Quadriceps_tendon.h5ad --seed ${s} --n-classes 22 > logs/raw_Quad_s${s}.log 2>&1 &
done
echo "raw-scGPT: 20 eval launched"
