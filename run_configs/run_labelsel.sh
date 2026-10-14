#!/bin/bash
cd ~/scREFINE/scREFINE_for_review_experiments
mkdir -p result_lsel logs_lsel
for ss in 1001 1002; do
  for s in 42 109; do
    nohup python3 run_scREFINE.py --data data/Brain.h5ad --output-dir result_lsel/Brain_ss${ss}_s${s} --labeled-ratio 0.05 --seed ${s} --split-seed ${ss} --epochs 100 --w-structure 0.1 > logs_lsel/Brain_ss${ss}_s${s}.log 2>&1 &
    nohup python3 run_scREFINE.py --data data/Fat.h5ad --output-dir result_lsel/Fat_ss${ss}_s${s} --labeled-ratio 0.05 --seed ${s} --split-seed ${ss} --epochs 100 --w-structure 0.1 > logs_lsel/Fat_ss${ss}_s${s}.log 2>&1 &
    nohup python3 run_scREFINE.py --data data/Retina.h5ad --output-dir result_lsel/Retina_ss${ss}_s${s} --labeled-ratio 0.05 --seed ${s} --split-seed ${ss} --epochs 100 --w-structure 0.1 > logs_lsel/Retina_ss${ss}_s${s}.log 2>&1 &
    nohup python3 run_scREFINE.py --data data/Breast_carcinoma.h5ad --output-dir result_lsel/Breast_ss${ss}_s${s} --labeled-ratio 0.05 --seed ${s} --split-seed ${ss} --epochs 100 --w-structure 0.1 > logs_lsel/Breast_ss${ss}_s${s}.log 2>&1 &
    nohup python3 run_scREFINE.py --data data/Quadriceps_tendon.h5ad --output-dir result_lsel/Quad_ss${ss}_s${s} --labeled-ratio 0.05 --seed ${s} --split-seed ${ss} --epochs 100 --w-structure 0.1 > logs_lsel/Quad_ss${ss}_s${s}.log 2>&1 &
  done
done
echo "Label-selection: 20 ta run launch hoye geche"