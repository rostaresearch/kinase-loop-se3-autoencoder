"""Merge v9 in-distribution latent + OOD addendum + lit-search rescue
into a single per-chain CSV.

Quality-control filter:
  - Drop chains with recon_rmsd > recon_max_rmsd (default 4.0 Å, which
    is well above in-dist 99-percentile of 3.1 Å).  This catches
    false-positive anchor matches (e.g. MIE in AURKA).
  - Keep all in-dist v9 chains regardless (these passed training-time QC).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v9-csv", required=True, type=Path,
                    help="v9_latent_with_labels.csv")
    ap.add_argument("--ood-csvs", nargs="+", required=True, type=Path,
                    help="OOD/litsearch/comprehensive addendum CSVs")
    ap.add_argument("--out-csv", required=True, type=Path)
    ap.add_argument("--recon-max-rmsd", type=float, default=5.5,
                    help="Reject OOD chains with recon RMSD above this. "
                         "v9 in-dist 99-pct is 3.1 Å.  Default 5.5 keeps "
                         "all canonical-anchor chains (including OOD "
                         "kinase families whose loops the v9 encoder "
                         "doesn't reproduce as tightly).")
    ap.add_argument("--allowed-anchors",
                    default="APE,SPE,ALE,PPE,PLE,SLE,TPE,TLE,AIE,AVE,AWE,"
                            "PTE,SWE,VPE,VLE",
                    help="Comma-separated list of accepted anchor motifs.  "
                         "MIE is explicitly EXCLUDED because it is a "
                         "verified false-positive helix-start match "
                         "(AURKA case study, see HTML report).  MYE and "
                         "YPE are excluded too -- no real kinase chains "
                         "use them in our dataset; they only appeared as "
                         "low-confidence matches.")
    ap.add_argument("--retrain-flag-recon", type=float, default=3.0,
                    help="Chains with recon RMSD above this are kept but "
                         "flagged in the 'needs_v9_1_retrain' column.")
    args = ap.parse_args()
    allowed_anchors = {a.strip() for a in args.allowed_anchors.split(",")}

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    # In-distribution v9
    v9 = pd.read_csv(args.v9_csv, keep_default_na=False)
    v9["ood"] = False
    v9["source"] = "v9_indist"
    v9["recon_rmsd"] = None
    v9["flank_rmsd_addendum"] = None
    v9["anchor_motif"] = ""
    print(f"v9 in-dist:        {len(v9)} chains")

    # OOD encodings (multiple CSVs; dedupe by chain_key, keeping the
    # one with lowest recon RMSD).
    keep_cols = ["chain_key", "gene", "group", "z0", "z1", "ood",
                 "dfg_spatial", "dihedral", "ligand_type",
                 "anchor_motif", "loop_length", "flank_rmsd",
                 "recon_rmsd"]
    all_ood = []
    for p in args.ood_csvs:
        df = pd.read_csv(p, keep_default_na=False)
        df["source"] = p.stem
        # only the cols we need + source
        df["chain_key"] = df["chain_key"].str.upper()
        all_ood.append(df[[c for c in keep_cols + ["source"]
                           if c in df.columns]])
        print(f"  OOD {p.name}:  {len(df)} chains "
              f"(min recon {df['recon_rmsd'].min():.2f}, "
              f"med {df['recon_rmsd'].median():.2f}, "
              f"max {df['recon_rmsd'].max():.2f})")

    ood = pd.concat(all_ood, ignore_index=True)
    ood = ood.sort_values("recon_rmsd").drop_duplicates(
        "chain_key", keep="first")
    print(f"OOD after dedupe: {len(ood)} chains")

    # Quality filter: keep only allowed anchors + recon RMSD ≤ threshold
    before = len(ood)
    ood_ok = ood[ood["anchor_motif"].isin(allowed_anchors)]
    n_dropped_anchor = before - len(ood_ok)
    print(f"After anchor filter (allowed: {sorted(allowed_anchors)}): "
          f"{len(ood_ok)} (dropped {n_dropped_anchor} non-canonical)")
    before = len(ood_ok)
    ood_ok = ood_ok[ood_ok["recon_rmsd"] <= args.recon_max_rmsd]
    print(f"After recon RMSD filter (<= {args.recon_max_rmsd} Å): "
          f"{len(ood_ok)} (dropped {before - len(ood_ok)} high-recon)")
    # Flag chains needing v9.1 retraining (high recon but valid anchor)
    ood_ok = ood_ok.copy()
    ood_ok["needs_v9_1_retrain"] = (ood_ok["recon_rmsd"] >
                                      args.retrain_flag_recon)
    print(f"  of which {ood_ok['needs_v9_1_retrain'].sum()} flagged "
          f"needs_v9_1_retrain (recon > {args.retrain_flag_recon} Å)")

    # Drop OOD chains that are ALSO in v9 in-distribution
    v9_keys = set(v9["chain_key"].str.upper())
    ood_ok = ood_ok[~ood_ok["chain_key"].isin(v9_keys)]
    print(f"After dropping chains already in v9 in-dist: {len(ood_ok)}")

    # Combine
    common_cols = sorted(set(v9.columns) | set(ood_ok.columns))
    for c in common_cols:
        if c not in v9.columns:
            v9[c] = ""
        if c not in ood_ok.columns:
            ood_ok[c] = ""
    merged = pd.concat([v9[common_cols],
                        ood_ok[common_cols]], ignore_index=True)
    merged.to_csv(args.out_csv, index=False)

    print()
    print(f"Wrote {args.out_csv}")
    print(f"Total chains: {len(merged)} "
          f"({len(v9)} in-dist + {len(ood_ok)} OOD)")
    print()
    print("Per-source breakdown:")
    print(merged["source"].value_counts().to_string())
    print()
    print("Per-gene chain counts (top 25):")
    print(merged.groupby("gene")["chain_key"].count().sort_values(
        ascending=False).head(25).to_string())


if __name__ == "__main__":
    main()
