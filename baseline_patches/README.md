# Baseline patches

These files are drop-in replacements/modifications that enforce the identical
evaluation protocol (same labeled cells from splits/*.npz, test-only evaluation).

- scSemiAAE_patched.py / scAGCL_patched.py — replace the corresponding files in the
  original scSemiAAE / scAGCL repositories.
- scgpt_*_patched.py — scGPT-backbone (common-embedding) variants.
- sccctr_eval.py — test-only evaluation of scCCTR's final iterative clustering output.
- scCCTR (original): edit main.py, line "kmeans = KMeans(n_clusters=num_clusters,
  init='k-means++')" → add ", random_state=42, n_init=10". No other change.