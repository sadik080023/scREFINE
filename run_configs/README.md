# Run configurations

All experiments use the four seeds 17, 42, 109, 172 and the splits in ../splits/.
scREFINE runs are CPU-only; launchers use nohup and can run in parallel.

| Script | Produces (paper) |
|---|---|
| run_all.sh + run_v2_rest.sh | Main table (Tables 3, 9; SI S4) — full model, 20 runs |
| run_raw.sh | Raw scGPT + K-means baseline (Table 3, SI S5) |
| run_loss_ablation.sh | Supervised loss ablation (Table 6, Fig 6, SI S6) — 100 runs |
| run_hardpl.sh | Hard pseudo-label variant (Table 3, SI S5) |
| run_labelsel.sh | Label-selection stability (Table 8) — 20 runs |
| run_lowlabel.sh | 2% soft vs hard guidance (Fig 10, SI S8) — 40 runs |
| run_wsens.sh | Structural-weight sensitivity (Fig 7, SI S9) — 60 runs |
| run_labelratio.sh | Label-ratio analysis (Fig 8, SI S3) — 60 runs |
| run_kesten.sh | Unsupervised K estimation (Table 7) |
| run_ksens.sh | Cluster-count sensitivity (Fig 9, SI S7) |
| launch_scgpt_semiaae.sh / launch_scgpt_agcl.sh / launch_scgpt_cctr.sh | Common-embedding baselines (Table 5, Fig 4, SI S2) |
| baselines/run_v2.sh (per-method folders) | Original scSemiAAE / scAGCL / scCCTR re-runs (Table 4, Fig 3, SI S1) |

All runs append to results/results_log_v2.csv.