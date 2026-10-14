#!/bin/bash
cd ~/scREFINE/scREFINE_for_review_experiments
mkdir -p result_hardpl logs_hardpl
for s in 17 42 109 172; do
  nohup python3 run_scREFINE.py --data data/Brain.h5ad --output-dir result_hardpl/Brain_s${s} --labeled-ratio 0.05 --seed ${s} --split-seed ${s} --epochs 100 --w-structure 0.1 --hard-pl > logs_hardpl/Brain_s${s}.log 2>&1 &
  nohup python3 run_scREFINE.py --data data/Fat.h5ad --output-dir result_hardpl/Fat_s${s} --labeled-ratio 0.05 --seed ${s} --split-seed ${s} --epochs 100 --w-structure 0.1 --hard-pl > logs_hardpl/Fat_s${s}.log 2>&1 &
  nohup python3 run_scREFINE.py --data data/Retina.h5ad --output-dir result_hardpl/Retina_s${s} --labeled-ratio 0.05 --seed ${s} --split-seed ${s} --epochs 100 --w-structure 0.1 --hard-pl > logs_hardpl/Retina_s${s}.log 2>&1 &
  nohup python3 run_scREFINE.py --data data/Breast_carcinoma.h5ad --output-dir result_hardpl/Breast_carcinoma_s${s} --labeled-ratio 0.05 --seed ${s} --split-seed ${s} --epochs 100 --w-structure 0.1 --hard-pl > logs_hardpl/Breast_s${s}.log 2>&1 &
  nohup python3 run_scREFINE.py --data data/Quadriceps_tendon.h5ad --output-dir result_hardpl/Quadriceps_tendon_s${s} --labeled-ratio 0.05 --seed ${s} --split-seed ${s} --epochs 100 --w-structure 0.1 --hard-pl > logs_hardpl/Quad_s${s}.log 2>&1 &
done
echo "Hard-PL: 20 ta run launch hoye geche"