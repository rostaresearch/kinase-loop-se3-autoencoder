# Reproduce Stage 1 — the 6,531-chain v9.1 loop dataset

**Verdict: nothing is missing.** Both files Yuxi flagged, and all the code that
generates them, are already in the consolidated bundle. This note pins down the
*canonical* route so the rebuild reproduces the published **6,531** chains
exactly, and explains the one trap that produces a different count.

`combined_v91_ca.pdb` (6,531 loops) is the shared input to **both** the
FoldingNet v9.1 model **and** the SE(3) distance-matrix model, so this stage is
the common prerequisite for any full rerun.

---

## The two files you were missing — where they actually are

Both exist in the bundle (self-contained authoritative copies are also in this
folder):

| file | canonical location in the bundle | rows | made by |
|------|----------------------------------|-----:|---------|
| `manifest_v9.csv` | `v9_release/manifest_v9.csv` | **5,318** | `build_v9_ca_spline.py` |
| `v9_addendum_merged.csv` | `manuscript_draft/data/v9_lgbm_shap/per_drug_analysis/v9_addendum_merged.csv` | **1,213** | `merge_all_encodings.py` |

`v9_addendum_merged.csv` holds only the OOD rescue chains, with a `source`
column: `v9_addendum_comprehensive` 689 + `v9_addendum_latent` 445 +
`v9_addendum_litsearch` 79 = 1,213.

**6,531 = 5,318 (in-distribution v9) + 1,213 (OOD rescue).**

They were hard to find because they live in two *different* sub-directories
(`v9_release/` and `.../per_drug_analysis/`), not co-located at the Stage-1 root.
This folder gathers them in one place.

---

## ⚠ The trap: `build_v91_stage1_all_in_one.py` gives 6,762, not 6,531

The self-contained `build_v91_stage1_all_in_one.py` rebuilds everything from raw
PDBs with a *looser* assignment-recovery path. It runs fine, but yields
**6,762** chains (5,603 in-dist + 1,159 OOD) — a superset that does **not** match
the published dataset. It is useful as an independent sanity check, **not** for
reproducing the canonical numbers. If you ran that and got 6,762, that is why the
intermediates looked "wrong / missing" — they weren't; the route was the looser one.

To reproduce the **canonical 6,531**, use the published intermediates below.
(Proof it works: `Pipeline/01_build_dataset/_work/rerun_full/` was built this way
and contains exactly 6,531.)

---

## Canonical reproduction (→ exactly 6,531)

Paths below assume the bundle root `$B` and Yuxi's PDB dir. Adjust if needed.

```bash
B=/data/student/yuxiz/v9_consolidated_v91
PDBS=/data/student/yuxiz/auto/Autoencoders/PDBs
python "$B/Pipeline/01_build_dataset/build_v91_dataset.py" \
  --merged-csv      "$B/manuscript_draft/data/v9_lgbm_shap/per_drug_analysis/v9_addendum_merged.csv" \
  --v9-combined-pdb "$B/v9_release/combined_v9_ca.pdb" \
  --v9-manifest     "$B/v9_release/manifest_v9.csv" \
  --v9-train-idx    "$B/v9_release/train_idx.txt" \
  --v9-test-idx     "$B/v9_release/test_idx.txt" \
  --pdb-dir         "$PDBS" \
  --ref-pdb         "$B/manuscript_draft/data/v9_lgbm_shap/figures/6UAN_full.pdb" \
  --out-dir         ./stage1_out
# -> stage1_out/combined_v91_ca.pdb  (6,531 MODELs)
#    stage1_out/manifest_v91.csv     (6,531 rows)
```

Verify: `grep -c '^MODEL' stage1_out/combined_v91_ca.pdb` → **6531**, and it
should be byte-for-chain identical to
`manuscript_draft/data/v91_lgbm_shap/combined_v91_ca.pdb`.

### If you also want to rebuild the two intermediates from raw (optional)

`manifest_v9.csv` (the 5,318 in-distribution chains) — from the original
DFG/APE assignments:

```bash
python build_v9_ca_spline.py \
  --assignments  <dfg_ape_assignments.csv>  \
  --full-pdb-dir "$PDBS" \
  --ref-pdb      6UAN_full.pdb --ref-chain C --ref-dfg 594 --ref-ape 623 \
  --out          ./v9_out          # writes combined_v9_ca.pdb + manifest_v9.csv
```

`v9_addendum_merged.csv` (the 1,213 OOD rescues) — encode the OOD chains, then
merge with QC:

```bash
# 1) produce the addendum CSVs (needs the v9 checkpoint best.ckpt):
python extract_and_encode_addendum.py  --checkpoint v9_release/best.ckpt ...  # comprehensive + latent
python extract_addendum_litsearch.py   --checkpoint v9_release/best.ckpt ...  # litsearch rescue
# 2) merge + QC-filter -> v9_addendum_merged.csv:
python merge_all_encodings.py \
  --v9-csv    v9_latent_with_labels.csv \
  --ood-csvs  v9_addendum_comprehensive.csv v9_addendum_latent.csv v9_addendum_litsearch.csv \
  --out-csv   v9_addendum_merged.csv
```

`merge_all_encodings.py` sets the `source` column and drops OOD chains with recon
RMSD above threshold and disallowed anchor motifs (MIE excluded — AURKA
false-positive). That QC is what makes the canonical OOD set 1,213.

---

## External raw inputs (all verified present on bohr, 2026-07-08)

| input | path | status |
|-------|------|--------|
| full per-chain PDBs (8,229) | `/data/student/yuxiz/auto/Autoencoders/PDBs` | ✓ |
| DFG/APE selection log | `/data/student/yuxiz/auto/Autoencoders/dfg_ape_selection_log.csv` | ✓ |
| v9 checkpoint | `v9_release/best.ckpt` | ✓ |
| KinCore FASTA | `manuscript_draft/data/kincore/PK_labels_PDB.fasta` | ✓ |
| BRAF 6UAN reference | `manuscript_draft/data/v9_lgbm_shap/figures/6UAN_full.pdb` | ✓ |

## Files in this kit

`manifest_v9.csv`, `v9_addendum_merged.csv`, `combined_v9_ca.pdb`,
`train_idx.txt`, `test_idx.txt` (the canonical inputs) + the five generating
scripts. Everything needed to run the canonical command above except the raw PDB
dir (too large — read it from Yuxi's `auto/Autoencoders/PDBs`).
