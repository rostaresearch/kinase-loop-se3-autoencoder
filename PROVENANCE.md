# Pipeline tree — raw data → results

Complete pathway for the kinase activation-loop **SE(3)** autoencoder (v9.1), with the
**expected count at every step** so you can assert you are not on stale data.
⚠ marks the only places old (BLAST/MUSTANG-era) pipeline data enters.

Everything lives in one shared, self-contained folder:
```
SHARE=/home/edina/kinase_v91_share        # world-readable; 5.6 GB; no other location needed
```
**Verified end-to-end 2026-07-16** — a clean rerun from these files reproduces R² = 0.915 on a single random chain-level split. See `code/grouped_cv.py` before quoting that number as generalisation: it falls to 0.859 (PDB-grouped) and 0.389 (gene-grouped).

---

```
╔═ STAGE 0 ─ RAW (externally re-downloadable) ══════════════════════════════════╗

  InterPro IPR011009 "Protein kinase-like domain" structure list
     $SHARE/data/raw/structure-matching-IPR011009.tsv .............. 8,223 entries
        │ cols: Accession, Source DB, Name, Experiment Type, Resolution, Chains, …
        ▼  RCSB  https://files.rcsb.org/download/{ID}.pdb
        │  ► reproduce_stage1/download_pdbs.py
        ▼
     $SHARE/data/PDBs/ {PDBID}.pdb ............................. 8,229 files (5.4 GB)
        ★ the pipeline only needs 4,533 of these (the manifest's unique PDB IDs):
            python download_pdbs.py --manifest-csv data/manifest_v91.csv --out-dir data/PDBs
          (--ipr-tsv = full 8,223 sweep · --check-only = audit what's missing)

  Kincore (Modi & Dunbrack)  $SHARE/data/raw/PK_labels_PDB.fasta ... DFG labels, gene, group
  BRAF reference             $SHARE/data/raw/6UAN_full.pdb ......... chain C; DFG=594, APE-E=623
                                                                     → loop = 594–623

⚠ LEGACY ROOT — the one place old-pipeline data enters
     $SHARE/data/raw/dfg_ape_selection_log.csv .................... 3,799 chains
        ⚠ from the ORIGINAL BLAST/MUSTANG pipeline; still has legacy_dfg_index /
          legacy_ape_index columns. The v9 chain list + DFG/APE index choices descend
          from this. cols incl: status, failure_reason, warning_reason, candidate_summary
╚═══════════════════════════════════════════════════════════════════════════════╝
                                     │
╔═ STAGE 1 ─ LOOP DATASET (6,531 chains) ══════════════════════════════════════╗
                                     ▼
  dfg_ape_selection_log ⚠  ──►  $SHARE/data/assignments_v9.csv ..... 5,318 rows
     (+ recover_dfgape_from_dismissed)   cols: status, chain_key,
                                               selected_dfg_resi, selected_ape_resi
        · `status` = outcome of DFG/APE anchor selection. Successes = "selected"
          (all 5,318 rows here); failures (no_dfg_permissive / no_ape_permissive /
          no_ape_after_dfg) stay behind in the log above.
        · build_v9_ca_spline.py filters `status == "selected"` — keep the column.
        │
        │ ► build_v9_ca_spline.py --assignments-csv
        │   flank Kabsch → BRAF 6UAN · cubic spline → fixed 27 Cα
        ▼
  $SHARE/data/v9_release/{combined_v9_ca.pdb, manifest_v9.csv} ..... 5,318 chains

  PDBs + Kincore + ⚠ v9 coordinate ckpt ($SHARE/data/v9_release/best.ckpt)
        │ ► extract_and_encode_addendum.py  (EGFR/ERBB/ALK … broadened anchors)
        │ ► extract_addendum_litsearch.py   (literature rescue)
        │ ► merge_all_encodings.py
        │   ⚠ OOD chains are QC-gated by the OLD **coordinate** model's recon RMSD
        │     (--recon-max-rmsd 5.5). So *which* OOD chains got in depends on the old
        │     model; the SE(3) model is trained on the result, it didn't choose it.
        ▼
  $SHARE/data/v9_addendum_merged.csv .............................. 1,213 OOD chains
        source: comprehensive 689 + latent 445 + litsearch 79
        │
        │ manifest_v9 (5,318) + addendum (1,213)  ► build_v91_dataset.py
        ▼
  $SHARE/data/{combined_v91_ca.pdb, manifest_v91.csv} ...... 6,531 chains ✅ CANONICAL

  ⚠ TWO ROUTES — only one is canonical:
     build_v91_stage1_all_in_one.py rebuilds from raw with looser recovery → **6,762**
     (5,603 + 1,159). A superset, NOT the published set. Use only as a sanity check.
     Proof the canonical route works: Pipeline/01_build_dataset/_work/rerun_full/ = 6,531
╚═══════════════════════════════════════════════════════════════════════════════╝
                                     │
╔═ STAGE 2 ─ CONSERVED SCAFFOLD (structural MSA) ══════════════════════════════╗
                                     ▼
  PDBs + BRAF 6UAN  ► map_v8_conserved_by_foldmason_chunks.py (FoldMason, 3Di alphabet)
        ⚠ "v8" in the filename is historical only — this IS the current v9.1 map.
        ▼
  $SHARE/data/v9_1_braf_mapped_conserved_residues.csv .. 794,867 rows / 6,531 chains
     cols: chain_key, pdb_id, chain, msa_column, braf_resi, pdb_resi, aa,
           three_di, mapping_source
     6UANC reference → **128** unique conserved non-loop residues, BRAF **452–718**
     covers 6,531/6,531 manifest chains (100%) · zero positions inside the loop
     594–623 → no leakage
╚═══════════════════════════════════════════════════════════════════════════════╝
                                     │
╔═ STAGE 3/4 ─ SE(3) AUTOENCODER + LATENT  (the current model) ════════════════╗
                                     ▼
  combined_v91_ca.pdb (6,531 × 27 Cα)
        │ ► q6_train_dm_ae.py — molearn CNN2d_AE, distance-matrix reconstruction,
        │   2-D latent, seed 25 (seeds 25/101/202 all ≈ equal)
        ▼
  $SHARE/q6_dm_ae_seed25.ckpt ....... 185 KB · 43,784 params · val DM-MSE 0.00123–0.00128
        │                              md5 eace79709ccbb0063f33a46fb7f220bf
        │ ► encode: coords → pairwise Cα–Cα distance matrix → encoder
        │   ⚠ standardisation MUST match training: global (x−89.87)/46.77 via molearn
        │     PDBData (never hand-rolled / per-chain — that bug inflated the latent ~75×)
        ▼
  $SHARE/v91_SE3_latent_seed25.csv ....... 6,531 chains · z0 −0.809…0.423 (unit scale)
        ⚠ NOT v91_full_kinome_CORRECT.csv / per_chain_selectivity_isolation.csv in old
          bundles — those are the superseded COORDINATE latent (|z0| up to 212).
╚═══════════════════════════════════════════════════════════════════════════════╝
                                     │
╔═ STAGE 5 ─ CONSERVED-DISTANCE PREDICTION + FEATURE IMPORTANCE ═══════════════╗
                                     ▼
  conserved map + manifest_v91 + PDBs + SE(3) latent
        │ ► predict_v9_lgbm_shap.py --ape-resi-floor 9999   ★ 9999 = FULL non-loop
        │   (624 = old N-lobe-only; keep only for the SI comparison)
        ▼
  128 residues → C(128,2) = 8,128 candidate pairs → **7,455** pass ≥75% coverage
  6,531 chains → 0 unmapped + 8 (>50% imputed) → **6,523** used (train 5,871 / test 652)
  ► **R² = 0.915**  (z0 0.939 · z1 0.892)  ← SINGLE RANDOM CHAIN-LEVEL SPLIT
  ► lgbm_residue_importance.csv · lgbm_shap_top.csv · lgbm_summary.csv

        │ ► grouped_cv.py  ★ REQUIRED before quoting any R² as generalisation
        ▼
  Same model, stricter validation (grouped_cv_results.csv):
      random 5-fold ......... 0.894 ± 0.005
      grouped by PDB entry .. 0.859 ± 0.010   (whole depositions held out)
      grouped by gene ....... 0.389 ± 0.123   (entire kinases held out)
  ► 0.915 is an INTERPOLATION figure. The PDB contains many near-duplicate chains of the
    same protein and same deposition, so a random chain split is pseudoreplicated.
    Use 0.859 for "unseen deposition" and 0.389 for "unseen kinase".

        │ ► eval_v9_fi_extended.py --ape-resi-floor 9999 → extended_fi_table.csv
        │ ► fi_methods_agreement.py / fi_methods_deeper.py → cross-method Spearman ρ
        │ ► top10_per_method.py → Table S1  (top features are CROSS-LOBE)
        │ ► dump_coverage_fullnonloop.py → per_residue_coverage.csv + pair_coverage.csv
        ▼
  residue-level ρ ≈ 0.86–0.94 (all 4 methods agree) · RF-impurity = pair-level outlier
╚═══════════════════════════════════════════════════════════════════════════════╝
                                     │
╔═ STAGE 6 ─ DOWNSTREAM BIOLOGY (all consume the SE(3) latent) ════════════════╗
   SE(3) latent ─┬─► v9_per_drug_latent · v9_drug_off_target · v9_selectivity_analysis
                 ├─► collect_v9_mutations → enumerate_v9_mutations
                 │      → test_mutation_significance → OncoKB join
                 │      284 populations tested · 16 significant (perm p<0.05)
                 │      ⚠ significance ≠ magnitude; screen constructs vs disease alleles
                 │        (results/oncokb_recheck_request.md)
                 └─► project_braf_md_se3 / FGFR2 → MD re-projection
╚═══════════════════════════════════════════════════════════════════════════════╝
                                     │
╔═ STAGE 7 ─ FIGURES ══════════════════════════════════════════════════════════╗
   stage-5/6 CSVs ─► figure_pipeline/run_all_figures.sh      (python compute)
                  ─► matlab/run_all_matlab_figs.m            (27 figs, 600 dpi PNG+PDF)
                  ▼  figures/
╚═══════════════════════════════════════════════════════════════════════════════╝
```

---

## Staleness checklist — assert before trusting any rerun

| checkpoint | expect | if different |
|---|---|---|
| latent `abs(z0).max()` | **≈0.81** | 212 ⇒ old COORDINATE model |
| `assignments_v9.csv` | 5,318 (all `selected`) | wrong assignment set |
| `v9_addendum_merged.csv` | 1,213 (689+445+79) | wrong addendum QC |
| `manifest_v91.csv` | **6,531** | 6,762 ⇒ all-in-one (non-canonical) route |
| conserved map, 6UANC ref | **128** residues, 452–718 | wrong/legacy map |
| candidate pairs | 8,128 | floor ≠ 9999 |
| features | **7,455** | 2,537 ⇒ floor 624 (N-lobe only) · 5,685 ⇒ incomplete map |
| chains used | **6,523** | check PDB coverage |
| **Combined R²** | **0.915** (single random split; 0.859 PDB-grouped, 0.389 gene-grouped — see `grouped_cv.py`) | see the superseded table below |

### Superseded numbers you may encounter
| number | what it was | why not to use |
|---|---|---|
| 0.889 | N-lobe only (2,537 feat), 5,683-chain subset | subset not reproducible; hides the C-lobe |
| 0.830 | full non-loop but an **incomplete** conserved map (5,685 feat, 6,459 chains) | that map missed 72 chains; it no longer exists |
| **0.915** ✅ | full non-loop + complete map (7,455 feat, 6,523 chains) | **current** |

## Known legacy inheritances (documented, deliberate)

1. **`dfg_ape_selection_log.csv`** — chain list + DFG/APE index choices for the 5,318
   in-distribution chains come from the old BLAST/MUSTANG pipeline.
2. **OOD addendum QC** — the 1,213 OOD chains were admitted using the **old coordinate
   model's** reconstruction RMSD as the gate, not the SE(3) model.
3. **Legacy filenames** — `v8_braf_mapped_conserved_residues.csv`,
   `conserved_residues_v9_schema.csv` are historical names.

None of these invalidate the SE(3) results (the model is trained and evaluated on the
resulting 6,531-chain set), but the **dataset composition is not yet independent of the old
pipeline**. A fully-clean rebuild would re-derive DFG/APE assignments and re-gate the
addendum with the SE(3) model — `build_v91_stage1_all_in_one.py` is the starting point.
