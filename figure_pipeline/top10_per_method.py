"""Table S1: top-10 conserved-distance features per importance method,
per latent target, from an extended_fi_table.csv."""
import argparse
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--extended-fi-csv", required=True, type=Path)
ap.add_argument("--out-csv", required=True, type=Path)
a = ap.parse_args()

d = pd.read_csv(a.extended_fi_csv)
methods = ["lgbm_gain", "lgbm_shap_meanabs", "lgbm_permutation", "rf_impurity"]
rows = []
for tgt in sorted(d["target"].unique()):
    sub = d[d["target"] == tgt]
    for m in methods:
        top = sub.sort_values(m, ascending=False).head(10)
        for rank, (_, r) in enumerate(top.iterrows(), 1):
            rows.append({"target": tgt, "method": m, "rank": rank,
                         "feature": r["feature"], "resi_i": int(r["resi_i"]),
                         "resi_j": int(r["resi_j"]), "importance": r[m]})
pd.DataFrame(rows).to_csv(a.out_csv, index=False)
print(f"wrote {a.out_csv} ({len(rows)} rows)")
