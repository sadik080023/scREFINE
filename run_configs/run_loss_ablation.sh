#!/bin/bash
# run_loss_ablation.sh — Reviewer #1 Major-3
cd ~/scREFINE/scREFINE_for_review_experiments
mkdir -p result_abl logs_abl

# Config F (Full) already done in main table -> skip. A-E run here.
# A: CE only | B: CE+Tri | C: CE+Proto | D: CE+Pair | E: CE+SupCon

run_cfg () {
  local ds=$1 seed=$2 tag=$3; shift 3
  nohup python3 run_scREFINE.py --data data/${ds}.h5ad \
    --output-dir result_abl/${ds}_${tag}_s${seed} \
    --labeled-ratio 0.05 --seed ${seed} --split-seed ${seed} \
    --epochs 100 --w-structure 0.1 "$@" \
    > logs_abl/${ds}_${tag}_s${seed}.log 2>&1 &
}

for s in 17 42 109 172; do
  for ds in Brain Fat Retina Breast_carcinoma Quadriceps_tendon; do
    run_cfg $ds $s A --w-triplet 0 --w-proto 0 --w-pairwise 0 --w-supcon 0 --w-ce 1.0
    run_cfg $ds $s B --w-triplet 1.0 --w-proto 0 --w-pairwise 0 --w-supcon 0 --w-ce 1.0
    run_cfg $ds $s C --w-triplet 0 --w-proto 0.5 --w-pairwise 0 --w-supcon 0 --w-ce 1.0
    run_cfg $ds $s D --w-triplet 0 --w-proto 0 --w-pairwise 0.1 --w-supcon 0 --w-ce 1.0
    run_cfg $ds $s E --w-triplet 0 --w-proto 0 --w-pairwise 0 --w-supcon 0.5 --w-ce 1.0
  done
done
echo "launched 100 loss-ablation runs (F = main table already)"
