#!/bin/bash
cd ~/scREFINE/scREFINE_for_review_experiments

for s in 17 42 109 172; do
  nohup python3 run_scREFINE.py --data data/Brain.h5ad            --output-dir result/Brain_s${s}            --labeled-ratio 0.05 --seed ${s} --split-seed ${s} --epochs 100 --w-structure 0.1 > logs/Brain_s${s}.log 2>&1 &
  nohup python3 run_scREFINE.py --data data/Fat.h5ad              --output-dir result/Fat_s${s}              --labeled-ratio 0.05 --seed ${s} --split-seed ${s} --epochs 100 --w-structure 0.1 > logs/Fat_s${s}.log 2>&1 &
  nohup python3 run_scREFINE.py --data data/Retina.h5ad           --output-dir result/Retina_s${s}           --labeled-ratio 0.05 --seed ${s} --split-seed ${s} --epochs 100 --w-structure 0.1 > logs/Retina_s${s}.log 2>&1 &
  nohup python3 run_scREFINE.py --data data/Breast_carcinoma.h5ad --output-dir result/Breast_carcinoma_s${s} --labeled-ratio 0.05 --seed ${s} --split-seed ${s} --epochs 100 --w-structure 0.1 > logs/Breast_carcinoma_s${s}.log 2>&1 &
  nohup python3 run_scREFINE.py --data data/Quadriceps_tendon.h5ad --output-dir result/Quadriceps_tendon_s${s} --labeled-ratio 0.05 --seed ${s} --split-seed ${s} --epochs 100 --w-structure 0.1 > logs/Quadriceps_tendon_s${s}.log 2>&1 &
done

echo "✅ 20 ta run launch hoye geche — PC shut down korleo remote host e cholbe"
