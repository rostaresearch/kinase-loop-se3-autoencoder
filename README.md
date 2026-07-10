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
| `WORKFLOW.md` | stage-by-stage guide to the whole pipeline |
| `figures/` | all publication figures (600-dpi PNG + vector PDF) |
| `figure_pipeline/` | one-command regeneration of every figure — `run_all_figures.sh` (Python compute) + `matlab/run_all_matlab_figs.m` + `README_figures.md` |
| `reproduce_stage1/` | how the 6,531-chain loop dataset is built (spline extraction + OOD addendum merge) |
| `results/` | key result CSVs — conserved-distance prediction, feature-importance agreement, top features, coverage, mutation significance |
| `code/` | core pipeline scripts (dataset → encode → conserved-distance prediction → drug / mutation analyses) |

## Key results (SE(3) latent, full non-loop conserved scaffold)
- Conserved-distance → loop-latent prediction **R² = 0.830** (z0 0.836 / z1 0.825), 128 non-loop residues (both lobes), 5,685 Cα–Cα distance pairs, 6,459 well-mapped chains.
- Top predictive distances are **cross-lobe** (αC↔αF 499–641; αEF↔αG 633–702) — the loop's position is read out by the relative arrangement of the two lobes.
- Four importance methods converge at the residue level (Spearman ρ 0.86–0.94).
- 16 activation-loop mutations shift significantly; 4 are OncoKB-curated drivers (BRAF V600E, EGFR T790M, FGFR2 K659N, FGFR3 V555M).
- Nearest-neighbour structural validation 99% closer than random; BRAF + FGFR2 MD ensembles re-projected through the SE(3) encoder.

## Reproducing figures
See `figure_pipeline/README_figures.md`. In short: point the paths at the SE(3)
latent + conserved map + PDB dir, run `bash run_all_figures.sh`, then
`run_all_matlab_figs.m` in MATLAB.
