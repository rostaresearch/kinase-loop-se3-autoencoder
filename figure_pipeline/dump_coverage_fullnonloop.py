"""Dump per-residue and per-pair structural coverage for the FULL non-loop
conserved-scaffold feature set (both lobes), for the conservation figure.
Reuses read_ca_map/norm_key from predict_v9_lgbm_shap so coverage matches
exactly how the LightGBM feature matrix is built."""
import sys, argparse
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "/home/edina/kinase_v4_training")
from predict_v9_lgbm_shap import read_ca_map, norm_key

ap = argparse.ArgumentParser()
ap.add_argument("--conserved-csv", required=True, type=Path)
ap.add_argument("--manifest-csv", required=True, type=Path)
ap.add_argument("--full-pdb-dir", required=True, type=Path)
ap.add_argument("--ref-chain-key", default="6UANC")
ap.add_argument("--ape-resi-floor", type=int, default=9999)  # 9999 = full non-loop
ap.add_argument("--out-dir", required=True, type=Path)
a = ap.parse_args()
a.out_dir.mkdir(parents=True, exist_ok=True)

cons = pd.read_csv(a.conserved_csv, keep_default_na=False)
cons["chain_key"] = norm_key(cons["chain_key"])
ref = cons[cons["chain_key"] == a.ref_chain_key.upper()].copy()
ref["pdb_resi"] = ref["pdb_resi"].astype(int)
ref = ref[ref["pdb_resi"] < a.ape_resi_floor]
resis = sorted(ref["pdb_resi"].unique().tolist())
print(f"{len(resis)} conserved non-loop residues: {resis[0]}..{resis[-1]}")
pairs = [(resis[i], resis[j]) for i in range(len(resis)) for j in range(i+1, len(resis))]
print(f"{len(pairs)} candidate pairs")

manifest = pd.read_csv(a.manifest_csv, keep_default_na=False)
manifest["chain_key"] = manifest["chain_key"].astype(str).str.upper()
chain_maps = (cons.groupby("chain_key")
              .apply(lambda g: dict(zip(g["braf_resi"].astype(int), g["pdb_resi"].astype(int))))
              ).to_dict()

n = len(manifest)
res_present = np.zeros((n, len(resis)), dtype=bool)
ridx = {r: k for k, r in enumerate(resis)}
for ii in range(n):
    if ii % 500 == 0: print(f"  chain {ii}/{n}")
    row = manifest.iloc[ii]
    cmap = chain_maps.get(row["chain_key"])
    if cmap is None: continue
    cas = read_ca_map(a.full_pdb_dir / f"{row['pdb']}.pdb", row["chain"])
    for r in resis:
        p = cmap.get(r)
        if p is not None and cas.get(p) is not None:
            res_present[ii, ridx[r]] = True

res_cov = res_present.mean(axis=0)
pd.DataFrame({"braf_resi": resis, "coverage": res_cov,
             "lobe": ["C-lobe" if r > 623 else "N-lobe" for r in resis]}
            ).to_csv(a.out_dir / "per_residue_coverage.csv", index=False)

pair_cov = np.empty(len(pairs), dtype=np.float32)
for k, (ri, rj) in enumerate(pairs):
    pair_cov[k] = (res_present[:, ridx[ri]] & res_present[:, ridx[rj]]).mean()
pd.DataFrame({"resi_i": [p[0] for p in pairs], "resi_j": [p[1] for p in pairs],
             "coverage": pair_cov}).to_csv(a.out_dir / "pair_coverage.csv", index=False)
print(f"pairs >=0.75 coverage: {(pair_cov>=0.75).sum()}/{len(pairs)}")
print("wrote per_residue_coverage.csv + pair_coverage.csv")
