"""R1: cache the RAW conserved-distance matrix (all candidate pairs, NaNs intact,
no coverage filter, no imputation) so preprocessing can be fit inside CV folds."""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, "/home/edina/kinase_v91_share/figure_pipeline")
from predict_v9_lgbm_shap import read_ca_map, norm_key

S = "/home/edina/kinase_v91_share"
OUT = Path("/home/edina/leakage"); OUT.mkdir(exist_ok=True)

conserved = pd.read_csv(f"{S}/data/v9_1_braf_mapped_conserved_residues.csv", keep_default_na=False)
conserved["chain_key"] = norm_key(conserved["chain_key"])
ref = conserved[conserved["chain_key"] == "6UANC"].copy()
ref["pdb_resi"] = ref["pdb_resi"].astype(int)
ref = ref[ref["pdb_resi"] < 9999]
braf = sorted(ref["pdb_resi"].unique().tolist())
pairs = [(braf[i], braf[j]) for i in range(len(braf)) for j in range(i + 1, len(braf))]
print(f"residues={len(braf)}  candidate pairs={len(pairs)}", flush=True)

manifest = pd.read_csv(f"{S}/data/manifest_v91.csv", keep_default_na=False)
manifest["chain_key"] = manifest["chain_key"].astype(str).str.upper()
cmaps = conserved.groupby("chain_key").apply(
    lambda g: dict(zip(g["braf_resi"].astype(int), g["pdb_resi"].astype(int)))).to_dict()

n = len(manifest)
X = np.full((n, len(pairs)), np.nan, dtype=np.float32)
for ii in range(n):
    if ii % 500 == 0:
        print(f"  chain {ii}/{n}", flush=True)
    row = manifest.iloc[ii]
    cmap = cmaps.get(row["chain_key"])
    if cmap is None:
        continue
    cas = read_ca_map(Path(f"{S}/data/PDBs") / f"{row['pdb']}.pdb", row["chain"])
    for jj, (ri, rj) in enumerate(pairs):
        pi, pj = cmap.get(ri), cmap.get(rj)
        if pi is None or pj is None:
            continue
        ci, cj = cas.get(pi), cas.get(pj)
        if ci is None or cj is None:
            continue
        X[ii, jj] = float(np.linalg.norm(ci - cj))

np.save(OUT / "X_raw.npy", X)
manifest.to_csv(OUT / "manifest_kept.csv", index=False)
pd.DataFrame(pairs, columns=["resi_i", "resi_j"]).to_csv(OUT / "candidate_pairs.csv", index=False)
print("saved", X.shape, "NaN frac %.4f" % np.isnan(X).mean(), flush=True)
