#!/bin/bash
# =====================================================================
# Full figure-generation pipeline — SE(3) distance-matrix model, v9.1
# Regenerates every conserved-distance / feature-importance figure and its
# underlying CSVs, using the FULL non-loop scaffold (both lobes, 128 residues,
# 5,685 pairs). Then hands off to MATLAB for the publication figures.
#
# Prereqs (Stage-1 dataset + AE encoding already done — see reproduce_stage1/):
#   - SE(3) 2-D latent per chain (chain_key,z0,z1[,labels])   -> $LAT
#   - conserved-residue FoldMason map                          -> $CONS
#   - manifest_v91.csv                                         -> $MAN
#   - full per-chain PDB dir ({PDBID}.pdb)                     -> $PDB
# Edit the paths below, then:  bash run_all_figures.sh
# =====================================================================
set -e

# ---- paths (EDIT THESE) ----
PY=${PY:-/home/edina/miniforge3/envs/kinase_ae/bin/python}
CODE=${CODE:-.}                                   # dir holding the .py scripts below
LAT=${LAT:?set LAT to the SE(3) latent CSV (chain_key,z0,z1)}
CONS=${CONS:?set CONS to v9_1_braf_mapped_conserved_residues.csv}
MAN=${MAN:?set MAN to manifest_v91.csv}
PDB=${PDB:?set PDB to the full per-chain PDB dir (PDBs_all)}
OUT=${OUT:-./figure_outputs}

# ---- the one methodological knob ----
# ape-resi-floor 9999  => FULL non-loop (N-lobe + C-lobe). The loop 594-623 is
# absent from the conserved map, so nothing leaks. Use 624 to reproduce the old
# N-lobe-only SI comparison.
FLOOR=${FLOOR:-9999}

mkdir -p "$OUT"
echo "[$(date)] === 1/5 conserved-distance LightGBM + SHAP (full non-loop) ==="
$PY "$CODE/predict_v9_lgbm_shap.py" --conserved-csv "$CONS" --manifest-csv "$MAN" \
    --full-pdb-dir "$PDB" --latent-csv "$LAT" --ape-resi-floor "$FLOOR" \
    --min-pair-coverage 0.75 --out "$OUT/lgbm_shap"

echo "[$(date)] === 2/5 extended feature-importance (gain/SHAP/perm/RF) ==="
$PY "$CODE/eval_v9_fi_extended.py" --conserved-csv "$CONS" --manifest-csv "$MAN" \
    --full-pdb-dir "$PDB" --latent-csv "$LAT" --ape-resi-floor "$FLOOR" \
    --min-pair-coverage 0.75 --out "$OUT/extended_fi"
FIT="$OUT/extended_fi/extended_fi_table.csv"

echo "[$(date)] === 3/5 cross-method agreement + within-method stability ==="
$PY "$CODE/fi_methods_agreement.py" --extended-fi-csv "$FIT" --out "$OUT/fi_methods_agreement"
$PY "$CODE/fi_methods_deeper.py"    --extended-fi-csv "$FIT" --out "$OUT/fi_methods_agreement" --n-boot 30 || true
$PY "$CODE/top10_per_method.py"     --extended-fi-csv "$FIT" --out-csv "$OUT/se3_top10_features_per_method.csv"

echo "[$(date)] === 4/5 scaffold coverage (for the conservation figure) ==="
$PY "$CODE/dump_coverage_fullnonloop.py" --conserved-csv "$CONS" --manifest-csv "$MAN" \
    --full-pdb-dir "$PDB" --ape-resi-floor "$FLOOR" --out-dir "$OUT/coverage"

echo "[$(date)] === 5/5 done — CSVs in $OUT. Now run the MATLAB figures: ==="
cat <<'NOTE'
  In MATLAB, from the matlab/ dir:
    COVDIR = '<OUT>/coverage';            % point at the coverage CSVs
    (copy lgbm_residue_importance.csv, lgbm_shap_top.csv, lgbm_test_predictions.csv,
     lgbm_summary.csv, fi_methods_agreement/*.csv into ../ so the fig_*.m find them)
    run('run_all_matlab_figs.m')          % regenerates every publication PNG/PDF
NOTE
echo "DONE_ALL_FIGURES_COMPUTE"
