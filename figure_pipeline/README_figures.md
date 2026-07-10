# Full figure-generation pipeline — SE(3) model (v9.1)

Regenerates every conserved-distance / feature-importance figure (and all the
downstream publication figures) from the SE(3) 2-D latent. The conserved-distance
analysis uses the **full non-loop scaffold** (both lobes, 128 residues, 5,685
Cα–Cα distance pairs); only the activation loop 594–623 is excluded as the target.

## Inputs (from earlier stages — see `../reproduce_stage1/`)
| input | what | example path |
|-------|------|--------------|
| SE(3) latent | per chain `chain_key,z0,z1[,labels]` | `v91_se3_out/per_chain_selectivity_isolation.csv` |
| conserved map | FoldMason BRAF-conserved residues | `v9_1_braf_mapped_conserved_residues.csv` |
| manifest | `manifest_v91.csv` | `v91_release/manifest_v91.csv` |
| PDB dir | full per-chain structures `{PDBID}.pdb` | `PDBs_all/` |

## Step 1 — compute (Python)
```bash
export PY=/path/to/env/bin/python  CODE=.  \
       LAT=.../per_chain_selectivity_isolation.csv  CONS=.../v9_1_braf_mapped_conserved_residues.csv \
       MAN=.../manifest_v91.csv  PDB=.../PDBs_all  OUT=./figure_outputs
bash run_all_figures.sh
```
Runs, in order:
1. `predict_v9_lgbm_shap.py --ape-resi-floor 9999` → LightGBM R²(z0,z1), SHAP, per-residue importance
2. `eval_v9_fi_extended.py --ape-resi-floor 9999` → gain / SHAP / permutation / RF per feature
3. `fi_methods_agreement.py` + `fi_methods_deeper.py` → cross-method Spearman ρ (feature + residue level) + bootstrap stability
4. `top10_per_method.py` → Table S1 (top-10 features per method)
5. `dump_coverage_fullnonloop.py` → per-residue + per-pair structural coverage (for the conservation figure)

**The one knob:** `FLOOR=9999` = full non-loop (main model). `FLOOR=624` reproduces
the old N-lobe-only SI comparison (R² 0.802 vs 0.830). The loop 594–623 is not in
the conserved map, so nothing leaks either way.

## Step 2 — figures (MATLAB)
Copy the CSVs the figure scripts expect into the report data dir (or symlink), then:
```matlab
cd matlab
COVDIR = '.../figure_outputs/coverage';
run('run_all_matlab_figs.m')     % → figures_matlab/*.png + *.pdf (600 dpi)
```
`run_all_matlab_figs.m` regenerates all 27 figures; each is independent and
error-isolated. Key new/updated scripts:
- `fig_residue_importance.m` — per-residue |SHAP|, full non-loop (455–710), loop gap shaded
- `fig_conservation_feature_selection.m` — coverage + 0.75 cutoff, both lobes
- `fig_fi_method_agreement.m` — cross-method ρ heatmap

## Scripts in this folder
`predict_v9_lgbm_shap.py`, `eval_v9_fi_extended.py`, `fi_methods_agreement.py`,
`fi_methods_deeper.py`, `top10_per_method.py`, `dump_coverage_fullnonloop.py`,
`run_all_figures.sh`, `matlab/` (all `fig_*.m` + `run_all_matlab_figs.m`).

## Note on the R² number
Full non-loop on the complete 6,459 well-mapped chains gives **R² = 0.830**
(z0 0.836 / z1 0.825). An earlier report quoted 0.889 on a smaller, better-resolved
5,683-chain subset that is not reproducible with the current inputs; the honest
full-set value is 0.830, and the +0.028 gain from adding the C-lobe is robust.
