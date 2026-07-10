# Kinase activation-loop autoencoder — workflow guide (v9.1)

A step-by-step guide to the pipeline, written so you can follow, reproduce, or
extend it. The canonical model is **v9.1** (6,531 kinase chains). Start by
opening the report:
`manuscript_draft/data/v91_lgbm_shap/v91_results_report.html`.

All scripts live in `code/pipeline_v3_no_mustang/`. Compute runs on **coulomb**
(GPU, conda env `kinase_ae`); the shared copy lives on **bohr** under
`/data/student/v9_consolidated/`.

---

## The idea in one paragraph

Each kinase structure has an *activation loop* whose conformation reports on
the enzyme's state. We extract that loop as a fixed-length 27-point Cα spline,
compress it with a small autoencoder (FoldingNet) to a **2-D latent**, and then
ask what that latent captures: does it separate DFG-in/out states, track
drug-binding mode, respond to phosphorylation and mutation, and can the loop
conformation be predicted from the rest of the (conserved) domain. The headline
finding is methodological — **preprocessing/representation choices dominate over
network architecture** — and the latent turns out to be a useful, interpretable
coordinate for kinase conformation.

---

## Pipeline, stage by stage

Each stage lists its **script**, **inputs → outputs**, and notes. Run them in
order; later stages consume earlier outputs.

### 1. Build the dataset (loops → fixed-length spline)
- **Script:** `build_v91_dataset.py` (calls `build_v9_ca_spline.py`)
- **In:** per-chain PDB files; the activation-loop anchor motifs.
- **Out:** `combined_v91_ca.pdb` (6,531 MODELs, 27 Cα each) + `manifest_v91.csv`.
- **Notes:** the loop is found between two conserved anchors — **DFG** (regex
  `D[FYLW]G`) and **APE** (`[APS][LP]E`). v9.1 *broadened* these from the
  original `D[FYL]G…[AS]PE`, which recovered 1,213 extra chains (EGFR `ALE`,
  ALK `PPE`, TYR/TKL/CMGC families) with no loss of accuracy. The loop Cα are
  spline-fit to a fixed 27 points so every chain has the same shape vector.

### 2. Structural multiple-sequence alignment (for the "conserved residues")
- **Script:** `map_v8_conserved_by_foldmason_chunks.py` (wraps FoldMason
  `easy-msa`)
- **In:** per-chain PDBs; BRAF 6UAN reference.
- **Out:** the script writes `v8_braf_mapped_conserved_residues.csv` (legacy
  name — keep it, the "v8" is historical); the v9.1 chains were merged into the
  canonical **`v9_1_braf_mapped_conserved_residues.csv`** used by all later
  stages — for every chain, which of its residues map to each conserved BRAF
  non-loop position.
- **Notes:** FoldMason aligns **3Di structural tokens**, not raw sequence —
  essential because cross-family sequence identity is too low for sequence MSA.
  This mapping is what lets us compute "the same" conserved Cα–Cα distance
  across all kinases. Coverage: 6,459 / 6,531 chains (98.9 %).
  The `--foldmason` arg defaults to a coulomb path; point it at your own
  FoldMason binary.

> **Note on running the scripts:** usage examples in some script docstrings show
> the original coulomb paths (`/tmp/...`, `/home/edina/...`) — substitute your
> own paths. All workflow scripts take explicit `--args`; nothing loads a
> hardcoded dataset. Two non-workflow helper scripts
> (`compute_backbone_recon_errors.py`, `predict_v8_top_features.py`) carry stale
> v6/v8 defaults — not part of this pipeline.

### 3. Train the autoencoder
- **Script:** `train_v9_ca_only.py`
- **In:** `combined_v91_ca.pdb`.
- **Out:** checkpoint `best.ckpt` (the v9.1 model).
- **Notes:** molearn FoldingNet `Small_AutoEncoder`, 2-D latent, 256 epochs.
  Validation MSE 0.0020 (better than the v9 base model's 0.0036).

### 4. Encode all chains → 2-D latent  ⚠ read the normalisation note
- **Script:** `embed_molearn_norm.py` (or the fixed `embed_v9_ca_only.py`)
- **In:** `combined_v91_ca.pdb` + `best.ckpt`.
- **Out:** `v91_full_kinome_CORRECT.csv` (chain_key, gene, z0, z1, labels).
- **⚠ CRITICAL LESSON:** inference **must** standardise coordinates exactly as
  training did — molearn's global `(coords − mean)/std` (mean 89.87, std 46.77),
  loaded via `PDBData` (so `fix_terminal` is applied). An earlier bug used
  per-chain centring with no std division; it silently produced a latent with
  ~75× the correct scale and wrecked every downstream number. If you re-encode
  anything, load through `PDBData`, not by hand.

### 5. Predict the latent from conserved non-loop distances
- **Script:** `predict_v9_lgbm_shap.py`
- **In:** conserved-residue map + PDBs + latent.
- **Out:** `lgbm_summary.csv`, SHAP/gain tables, `figures/lgbm_*`.
- **Result:** LightGBM R² = **0.902** (z0 0.903, z1 0.901) — the loop
  conformation is strongly determined by the surrounding conserved scaffold;
  the relationship is non-linear (linear models trail badly).
- **⚠ LESSON — coverage filter:** pass `--max-imputed-frac 0.5`. Chains that
  have a FoldMason entry but whose residues don't actually cover the conserved
  pairs become mostly mean-imputed and collapse onto the centroid, forming a
  spurious horizontal "band" in the predicted-vs-actual plot. The filter drops
  those (848 chains) and keeps 5,683 well-covered ones.

### 6. Feature-importance robustness (which residues matter, how robustly)
- **Scripts:** `eval_v9_fi_extended.py` → `fi_methods_agreement.py`,
  `fi_methods_deeper.py`; `compare_ca_vs_minatom.py`
- **Result:** four importance methods (LightGBM gain, SHAP, permutation, RF)
  agree at the residue level (ρ ≥ 0.8); the consensus set is αC / hinge / αE /
  catalytic-loop / pre-DFG. Distance metric (Cα–Cα vs min heavy-atom) doesn't
  change downstream R². (Ridge is deliberately *not* used — wrong model class
  for this non-linear problem.)

### 7. Drug-bound chains, off-target, selectivity
- **Scripts:** `v9_per_drug_latent.py`, `v9_drug_off_target.py`,
  `v9_drug_offtarget_vs_literature.py`, `v9_selectivity_analysis.py`
- **Out:** `per_drug_analysis/`, `per_drug_off_target_table.csv`,
  `off_target_vs_literature/`, `per_gene_compactness.csv`, etc.
- **Notes:** 24 FDA inhibitors mapped by Kincore ligand code; Type 1/2/allosteric
  fall in the expected DFG regions. **Honest negative result:** off-target
  latent separation does **not** predict Davis 2011 selectivity (Spearman +0.09,
  n=6) — an earlier "+0.49" was an artifact of the mis-normalised latent.

### 8. Phosphorylation worked example
- **Script:** `figure_map2k1_phospho_example.py`
- **Out:** `map2k1_phospho_example.png`. MAP2K1 phospho-mimetic/phospho-dead
  mutants separate from WT in the latent.

### 9. Mutation validation
- **Scripts:** `collect_v9_mutations.py` → `enumerate_v9_mutations.py` →
  `test_mutation_significance.py` → `join_oncokb_to_skeleton.py`
- **Out:** `v91_mutation_validation_skeleton.csv`, `v91_significance_summary.csv`,
  `v91_mutation_validation_with_oncokb.csv`.
- **Result:** 284 PDB-annotated mutation populations tested (permutation p +
  Mahalanobis σ vs WT scatter); **26 significant at p < 0.05** (e.g. EGFR T790M).

### 10. Extended biology
- **Script:** `v9_extended_analyses.py`
- **Out:** `extended/` — within-kinase diversity, novel latent regions, NN
  validation, ABL1 inhibitor-escape routes.

### 11. MD validation (does the latent generalise to dynamics?)
- **Script:** `project_braf_md_to_v9.py`
- **In:** Clayton/Shen 2025 BRAF V600E MD trajectories (27 runs).
- **Out:** `md_projection/` + `figures/braf_md_density_on_v91_latent.png`.
- **Result:** 210,002 MD frames; **87.5 % land inside the BRAF experimental
  region** of the latent — the embedding generalises to physically-sampled
  conformations. (Project MD with the *same* standardisation as training; see §4.)

### 12. Ablations + robustness (methodology)
- **Scripts:** `train_v9_variant.py` (×8 configs) → `evaluate_v9_variants.py`
- **Out:** `v91_ablation_variant_summary.csv`, `v91_ablation_probes.csv`.
- **Results:** seed-robust (aligned reconstruction RMSD 2.3–2.5 Å across seeds);
  the v9 "AdamW weight-decay over-regularises" effect does **not** replicate at
  v9.1's larger dataset size. **Limitation:** when whole kinase families are
  held out, reconstruction degrades (aligned 9.7 Å median) — the model is
  *family-interpolative, not family-extrapolative*. Use it within the
  kinase-fold distribution it was trained on.
- **⚠ LESSON:** reconstruction RMSD must be **Kabsch-superposed** per chain;
  raw (unaligned) RMSD penalises placement and grossly overstates error.

---

## Key files (in `manuscript_draft/data/v91_lgbm_shap/`)

| File | What it is |
|------|------------|
| `v91_results_report.html` / `.pdf` | **Start here** — the consolidated report |
| `v91_full_kinome_CORRECT.csv` | the canonical 6,531-chain 2-D latent + labels |
| `combined_v91_ca.pdb` / `manifest_v91.csv` | the loop spline inputs |
| `lgbm_summary.csv`, `extended_fi_table.csv` | conserved-distance prediction + FI |
| `per_drug_analysis/`, `off_target_vs_literature/` | drug analyses |
| `v91_mutation_validation_with_oncokb.csv` | mutation validation + OncoKB |
| `md_projection/` | BRAF MD projected onto the latent |
| `v91_ablation_variant_summary.csv` | ablation / seed / holdout results |

The trained **v9.1 model** is in the bundle at
`manuscript_draft/data/v91_lgbm_shap/v91_best.ckpt`. To load it and encode a loop:

```python
import torch
from molearn.data import PDBData
from molearn.models.small_foldingnet import Small_AutoEncoder

data = PDBData(); data.import_pdb(".../v91_lgbm_shap/combined_v91_ca.pdb")
data.fix_terminal(); data.atomselect(atoms=["CA"]); data.prepare_dataset()  # global (x-mean)/std
net = Small_AutoEncoder(out_points=27)
st = torch.load(".../v91_lgbm_shap/v91_best.ckpt", map_location="cpu", weights_only=False)
net.load_state_dict(st.get("model_state_dict", st.get("state_dict", st))); net.eval()
with torch.no_grad():
    z = net.encode(data.dataset.float())   # (N, 2) latent == v91_full_kinome_CORRECT.csv
```
(molearn 2.0.1 wants `(B,3,N)` input and `PDBData` already returns that; 2.0.4 wants
`(B,N,3)` — numerics match either way. **Always load via `PDBData`** so the
standardisation matches training — see gotcha #1.)

---

## Five gotchas that cost us time (so they don't cost you)

1. **Inference normalisation must match training** (§4). Load through molearn
   `PDBData`; never hand-roll `(coords−mean)/std` or per-chain centring.
2. **Filter mostly-imputed chains** before the conserved-distance prediction
   (§5, `--max-imputed-frac 0.5`) or you get a centroid-collapse band.
3. **Kabsch-align before measuring reconstruction RMSD** (§12); raw RMSD is
   inflated and misleading.
4. **The latent scale is arbitrary** for an unregularised autoencoder — what
   matters is that the latent is smooth and downstream-predictable, not its
   absolute coordinates. Don't compare raw latent coordinates across separately
   trained models.
5. **The model doesn't extrapolate to unseen kinase families** — keep
   applications within the trained fold distribution.
