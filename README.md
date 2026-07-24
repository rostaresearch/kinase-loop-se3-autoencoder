# Kinase activation-loop autoencoder (SE(3)-invariant) — v9.1

A 2-D interpretable latent for the kinase **activation loop**, learned with an
**SE(3)-invariant distance-matrix autoencoder** (molearn `CNN2d_AE`) over 6,531
human kinase chains. The latent separates DFG-in/out states, organises inhibitors
by binding mode, responds to phosphorylation and disease mutations, and — the
headline methodological result — the loop conformation is **predictable from the
surrounding conserved scaffold** (both lobes) at R² = 0.83.

This repository holds the **figures and code** to reproduce them. The written
manuscript/report is **not** included here (it is shared separately while
unpublished). It also excludes large data (per-chain PDBs, model checkpoints, MD
trajectories), tokens, and unrelated analyses.

## Layout
| dir | contents |
|-----|----------|
| **`PROVENANCE.md`** | **the pipeline tree — raw data → results, with the expected count at every step and the legacy-inheritance flags. Start here.** |
| `WORKFLOW.md` | stage-by-stage guide (older; see PROVENANCE.md for the current tree) |
| `figures/` | all publication figures (600-dpi PNG + vector PDF) |
| `figure_pipeline/` | one-command regeneration of every figure — `run_all_figures.sh` (Python compute) + `matlab/run_all_matlab_figs.m` + `README_figures.md` |
| `reproduce_stage1/` | how the 6,531-chain loop dataset is built (spline extraction + OOD addendum merge) |
| `results/` | key result CSVs — conserved-distance prediction, feature-importance agreement, top features, coverage, mutation significance |
| `code/` | core pipeline scripts (dataset → encode → conserved-distance prediction → drug / mutation analyses) |

## Key results (SE(3) latent, full non-loop conserved scaffold)
- Conserved-distance → loop-latent prediction **R² = 0.915** (z0 0.939 / z1 0.892) on a **single random chain-level split**, from **128** non-loop residues spanning **both lobes** → **7,455** Cα–Cα distance pairs, **6,523** chains. *(Verified end-to-end 2026-07-16.)*
- **Read 0.915 as interpolation, not generalisation.** Under grouped cross-validation
  (`code/grouped_cv.py`, `results/grouped_cv_results.csv`) accuracy degrades monotonically as the
  grouping tightens: random 5-fold **0.894 ± 0.005** → grouped by PDB entry **0.859 ± 0.010** →
  **grouped by gene 0.389 ± 0.123**. The scaffold→loop relationship is real (whole depositions held
  out still gives 0.859) but a substantial component of it is **kinase-specific rather than
  transferable**. Quote the gene-grouped number for "predict a novel kinase".
- **Preprocessing leakage is not the problem.** Refitting the ≥75% pair-coverage filter and the
  imputation means inside each training fold (`code/nested_cv.py`, `results/nested_cv_results.csv`)
  changes R² by −0.001 (random), +0.004 (PDB-grouped) and −0.015 (gene-grouped) — negligible and of
  inconsistent sign. The validation-*design* effect above is what matters.
- **Latent axes are not identifiable across seeds** (`code/procrustes_seeds.py`,
  `results/seed_*.csv`). Five models with equivalent reconstruction error (val DM-MSE
  0.00116–0.00124): the raw axes correlate at only |r| 0.88 / 0.80 (worst pair 0.54), rising to
  0.96 / 0.94 after orthogonal Procrustes alignment. **6 of 10 model pairs need a reflection** to
  align — the expected signature of a distance-based (E(3)-invariant) representation. The
  *configuration* is reproducible (rotation-invariant ρ = 0.937).
- Top predictive distances are **cross-lobe**. Only two importance statements replicate across all
  five seeds: **αC↔αF 499–641 is rank 1 for the leading coordinate** (in every seed, under every
  method), and at residue level the top-10 core is 499/575/633/641 (leading) and **574/632/633 —
  the αEF region** (second). *No feature is rank 1 for the second coordinate in more than one seed*,
  so the single-model result "525–577 dominates z1" is method-unanimous but **not** seed-robust and
  must not be quoted. Per-residue importance agrees across seeds at ρ = 0.79 / 0.70.
- Four importance methods converge at the residue level (Spearman ρ 0.76–0.95); at the individual-pair
  level the three LightGBM-derived rankings agree (ρ 0.89–0.96) and RF impurity is the outlier (0.24–0.28).
- 16 activation-loop mutations shift at uncorrected permutation p<0.05, of which **7 survive
  Benjamini–Hochberg q<0.05** across the 82 testable populations; 4 are OncoKB-curated drivers
  (BRAF V600E, EGFR T790M, FGFR2 K659N, FGFR3 V555M).
- Because the encoder consumes only interatomic distances, the invariance group is the **full
  Euclidean group E(3)** (rotation, translation *and reflection*), not SE(3); the "SE(3)" label is
  retained only for continuity with the model name.
- Nearest-neighbour structural validation 99% closer than random; BRAF + FGFR2 MD ensembles re-projected through the SE(3) encoder.

## Reproducing figures
See `figure_pipeline/README_figures.md`. In short: point the paths at the SE(3)
latent + conserved map + PDB dir, run `bash run_all_figures.sh`, then
`run_all_matlab_figs.m` in MATLAB.

**Raw data:** `reproduce_stage1/download_pdbs.py` fetches the structures from RCSB.
The pipeline needs only the **4,533** PDB entries the manifest references (not the full
~8,200-entry InterPro sweep):
```bash
python reproduce_stage1/download_pdbs.py --manifest-csv manifest_v91.csv --out-dir PDBs
python reproduce_stage1/download_pdbs.py --manifest-csv manifest_v91.csv --out-dir PDBs --check-only  # audit
```

> **Note on `figures/` and `results/`:** these were generated from an earlier run whose
> conserved-residue map was incomplete (5,685 features / 6,459 chains → R² 0.830). The map
> has since been corrected (7,455 features / 6,523 chains → **R² 0.915**) and these outputs
> are being regenerated. `PROVENANCE.md` lists every superseded number and why.
