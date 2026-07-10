"""Drug-design selectivity analysis on the v9 AE latent.

Three quantitative views on how easy it is to design a selective drug:

  (A) Per-kinase compactness: how broad is one kinase's latent footprint?
      Compact targets are intrinsically easier to drug selectively;
      diffuse targets sample many states and risk drug-resistance via
      state-switching.

  (B) Selectivity isolation index: per chain (and aggregated per gene),
      the latent distance to the nearest non-self kinase chain. High
      isolation = drug poses for this kinase are unlikely to cross-react.

  (C) Ligand-type latent footprints: confirms Type 1 / Type 2 / Allosteric
      ↔ DFG-state correspondence and shows which kinases accept which
      inhibitor class.

Inputs:
  --latent-csv      v9_latent_with_labels.csv (has z0, z1, gene, group,
                    dfg_spatial, dihedral, ligand_type, chain_key)
  --out             output directory

Outputs:
  per_gene_compactness.csv          (A)
  per_chain_selectivity_isolation.csv  (B)
  per_gene_selectivity_isolation.csv   (B aggregated)
  per_ligand_type_summary.csv       (C)
  drug_design_targets_table.csv     joint (A + B) ranked table

Figures (under <out>/figures/):
  compactness_histogram.png
  compactness_ranked_bars.png
  selectivity_isolation_scatter.png
  compactness_vs_isolation_scatter.png
  ligand_type_latent_facets.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import adjusted_mutual_info_score

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

DFG_COLORS = {
    "DFGin": "#4c78a8", "DFGout": "#f58518",
    "DFGinter": "#54a24b", "None": "#bbbbbb",
}
LIG_COLORS = {
    "Type1":         "#4c78a8",
    "Type2":         "#f58518",
    "Type1.5_Back":  "#9c6dc3",
    "Type1.5_Front": "#a072cd",
    "Type3":         "#e45756",
    "Allosteric":    "#54a24b",
    "ATPlike":       "#b79a20",
    "No_ligand":     "#bbbbbb",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent-csv", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--min-chains-per-gene", type=int, default=10,
                    help="Minimum chains a gene needs to be included in compactness.")
    ap.add_argument("--n-neighbours", type=int, default=5,
                    help="Number of nearest non-self chains for the isolation index.")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out / "figures"; fig_dir.mkdir(exist_ok=True)

    df = pd.read_csv(args.latent_csv, keep_default_na=False)
    df["z0"] = pd.to_numeric(df["z0"], errors="coerce")
    df["z1"] = pd.to_numeric(df["z1"], errors="coerce")
    df = df[df["z0"].notna() & df["z1"].notna()].copy()
    df["gene"] = df["gene"].astype(str)
    print(f"Loaded {len(df)} chains, {df['gene'].nunique()} unique genes")
    Z = df[["z0", "z1"]].to_numpy()

    # ---------- (A) per-kinase compactness ----------
    print("\n(A) Per-kinase compactness")
    rows = []
    for gene, sub in df.groupby("gene"):
        n = len(sub)
        if n < args.min_chains_per_gene: continue
        pts = sub[["z0", "z1"]].to_numpy()
        centroid = pts.mean(0)
        dists = np.linalg.norm(pts - centroid, axis=1)
        cov = np.cov(pts.T) if n > 1 else np.eye(2)
        ellipse_area_95 = float(np.pi * 5.991 * np.sqrt(max(np.linalg.det(cov), 0)))
        rows.append({
            "gene": gene, "n_chains": int(n),
            "centroid_z0": float(centroid[0]),
            "centroid_z1": float(centroid[1]),
            "median_radius_from_centroid": float(np.median(dists)),
            "p95_radius_from_centroid": float(np.quantile(dists, 0.95)),
            "ellipse_area_95pct": ellipse_area_95,
            "trace_cov": float(np.trace(cov)),
        })
    compactness = pd.DataFrame(rows).sort_values("p95_radius_from_centroid")
    compactness.to_csv(args.out / "per_gene_compactness.csv", index=False)
    print(f"  {len(compactness)} genes with >={args.min_chains_per_gene} chains")
    print(f"  Tightest (top 10):")
    print(compactness.head(10)[["gene","n_chains","p95_radius_from_centroid","ellipse_area_95pct"]].to_string(index=False))
    print(f"  Most diffuse (bottom 10):")
    print(compactness.tail(10)[["gene","n_chains","p95_radius_from_centroid","ellipse_area_95pct"]].to_string(index=False))

    # ---------- (B) per-chain selectivity isolation ----------
    print("\n(B) Selectivity isolation index")
    # For each chain, mean distance to N nearest chains of OTHER genes.
    gene_arr = df["gene"].to_numpy()
    iso = np.full(len(df), np.nan)
    # Build a KDTree of all chains; for each chain, find non-self neighbours.
    tree = cKDTree(Z)
    for i in range(len(df)):
        # Query enough neighbours to ensure we get N non-self ones.
        k_query = min(len(df), args.n_neighbours * 20 + 5)
        dists, idxs = tree.query(Z[i], k=k_query)
        own_gene = gene_arr[i]
        non_self_d = [d for d, j in zip(dists[1:], idxs[1:])
                      if gene_arr[j] != own_gene]
        if len(non_self_d) >= args.n_neighbours:
            iso[i] = float(np.mean(non_self_d[:args.n_neighbours]))
    df["selectivity_isolation"] = iso
    df[["chain_key", "gene", "group", "dfg_spatial", "ligand_type",
        "z0", "z1", "selectivity_isolation"]].to_csv(
        args.out / "per_chain_selectivity_isolation.csv", index=False)

    # Per-gene aggregation.
    per_gene_iso = (df.groupby("gene")["selectivity_isolation"]
                    .agg(["median", "mean", "std", "count"])
                    .reset_index())
    per_gene_iso = per_gene_iso[per_gene_iso["count"] >= 5].sort_values(
        "median", ascending=False)
    per_gene_iso.to_csv(args.out / "per_gene_selectivity_isolation.csv",
                        index=False)
    print(f"  Top 10 most isolated genes (selective targets):")
    print(per_gene_iso.head(10).to_string(index=False))
    print(f"  Bottom 10 least isolated (crowded in latent):")
    print(per_gene_iso.tail(10).to_string(index=False))

    # ---------- Joint compactness × isolation table ----------
    joint = compactness.merge(per_gene_iso.rename(
        columns={"median": "iso_median", "mean": "iso_mean",
                 "std": "iso_std", "count": "iso_count"}),
        on="gene", how="inner")
    joint["target_score"] = (
        joint["iso_median"] / joint["p95_radius_from_centroid"].replace(0, np.nan)
    )
    joint = joint.sort_values("target_score", ascending=False)
    joint.to_csv(args.out / "drug_design_targets_table.csv", index=False)
    print(f"\nTop 15 'easiest selective target' candidates "
          "(high isolation / low own spread):")
    print(joint.head(15)[["gene","n_chains","p95_radius_from_centroid",
                          "iso_median","target_score"]].to_string(index=False))

    # ---------- (C) Ligand-type latent footprints ----------
    print("\n(C) Ligand-type latent footprints")
    df["ligand_type_clean"] = df["ligand_type"].astype(str).replace("", "No_ligand")
    lig_counts = df["ligand_type_clean"].value_counts()
    print(lig_counts.to_string())

    # Per-ligand AMI vs DFG state.
    lig_rows = []
    for lig, sub in df.groupby("ligand_type_clean"):
        if len(sub) < 20: continue
        valid = sub[sub["dfg_spatial"].isin(["DFGin", "DFGout", "DFGinter"])]
        if len(valid) < 10: continue
        labels = valid["dfg_spatial"].astype(str).values
        # Trivial AMI = perfect labeling against itself, so we use latent
        # k-means clustering vs labels.
        from sklearn.cluster import KMeans
        nclust = max(3, len(set(labels)))
        cl = KMeans(n_clusters=nclust, random_state=25, n_init=10).fit(
            valid[["z0", "z1"]].to_numpy())
        ami = float(adjusted_mutual_info_score(labels, cl.labels_))
        lig_rows.append({
            "ligand_type": lig, "n_chains": int(len(sub)),
            "median_z0": float(sub["z0"].median()),
            "median_z1": float(sub["z1"].median()),
            "dfg_in_frac": float((sub["dfg_spatial"]=="DFGin").mean()),
            "dfg_out_frac": float((sub["dfg_spatial"]=="DFGout").mean()),
            "dfg_inter_frac": float((sub["dfg_spatial"]=="DFGinter").mean()),
            "ami_kmeans_vs_dfg": ami,
        })
    lig_df = pd.DataFrame(lig_rows).sort_values("n_chains", ascending=False)
    lig_df.to_csv(args.out / "per_ligand_type_summary.csv", index=False)
    print(lig_df.to_string(index=False))

    # ---------- FIGURES ----------

    # Compactness histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(compactness["p95_radius_from_centroid"], bins=40,
            color="#315f8e", alpha=0.75, edgecolor="white")
    ax.axvline(compactness["p95_radius_from_centroid"].median(),
               color="grey", ls="--",
               label=f"median: {compactness['p95_radius_from_centroid'].median():.0f}")
    ax.set_xlabel("p95 latent radius from per-gene centroid")
    ax.set_ylabel(f"genes (n={len(compactness)})")
    ax.set_title("(A) Per-kinase conformational compactness\n"
                 f"genes with ≥{args.min_chains_per_gene} PDB chains")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "compactness_histogram.png")
    plt.close(fig)

    # Top + bottom 15 compact genes
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True)
    for ax, sub, title in [
        (axes[0], compactness.head(15),
         f"Most compact (easiest selective target)"),
        (axes[1], compactness.tail(15)[::-1],
         f"Most diffuse (hardest selective target)"),
    ]:
        ax.barh(range(len(sub)), sub["p95_radius_from_centroid"],
                color="#315f8e" if "compact" in title else "#c0504d")
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels([f"{g} (n={n})"
                            for g, n in zip(sub["gene"], sub["n_chains"])])
        ax.set_xlabel("p95 latent radius")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(fig_dir / "compactness_ranked_bars.png")
    plt.close(fig)

    # Selectivity isolation scatter
    fig, ax = plt.subplots(figsize=(8, 7))
    valid = df["selectivity_isolation"].notna()
    sc = ax.scatter(df.loc[valid, "z0"], df.loc[valid, "z1"],
                    c=df.loc[valid, "selectivity_isolation"],
                    cmap="viridis", s=5, alpha=0.7, edgecolor="none",
                    vmin=0,
                    vmax=df["selectivity_isolation"].quantile(0.95))
    fig.colorbar(sc, ax=ax,
                 label=f"mean dist to {args.n_neighbours} nearest non-self chains")
    ax.set_xlabel("z0"); ax.set_ylabel("z1")
    ax.set_title("(B) Selectivity isolation: per-chain latent distance\n"
                 "to the nearest non-self kinase chains")
    fig.tight_layout()
    fig.savefig(fig_dir / "selectivity_isolation_scatter.png")
    plt.close(fig)

    # Compactness × isolation joint scatter (per gene)
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(joint["p95_radius_from_centroid"], joint["iso_median"],
                    s=joint["n_chains"]*2, alpha=0.65, c="#315f8e",
                    edgecolor="black", linewidth=0.5)
    # Annotate top 15 by target_score
    for _, r in joint.head(15).iterrows():
        ax.annotate(r["gene"], (r["p95_radius_from_centroid"], r["iso_median"]),
                    fontsize=9, ha="left", va="bottom",
                    xytext=(3, 3), textcoords="offset points",
                    color="#1b4870")
    # And bottom 5 (worst targets) in red
    for _, r in joint.tail(5).iterrows():
        ax.annotate(r["gene"], (r["p95_radius_from_centroid"], r["iso_median"]),
                    fontsize=9, ha="left", va="top",
                    xytext=(3, -3), textcoords="offset points",
                    color="#c0504d")
    ax.set_xlabel("Own latent spread (p95 radius from centroid)")
    ax.set_ylabel("Median isolation distance to other kinases")
    ax.set_title("(A × B) Drug-design selectivity landscape per kinase\n"
                 "compact + isolated (top-left) = easiest selective targets;\n"
                 "diffuse + crowded (bottom-right) = hardest")
    fig.tight_layout()
    fig.savefig(fig_dir / "compactness_vs_isolation_scatter.png")
    plt.close(fig)

    # Ligand-type latent facets
    main_ligs = [lig for lig, n in lig_counts.items()
                 if n >= 50 and lig != "No_ligand"]
    n_lig = len(main_ligs)
    ncol = 3
    nrow = (n_lig + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5*ncol, 4*nrow),
                              squeeze=False, sharex=True, sharey=True)
    for k, lig in enumerate(main_ligs):
        ax = axes[k // ncol, k % ncol]
        # Background: all chains greyed
        ax.scatter(df["z0"], df["z1"], s=2, alpha=0.1, color="#dddddd",
                   edgecolor="none")
        # Ligand-bound chains coloured by DFG state
        sub = df[df["ligand_type_clean"] == lig]
        for cls, color in DFG_COLORS.items():
            m = sub["dfg_spatial"] == cls
            if m.sum():
                ax.scatter(sub.loc[m, "z0"], sub.loc[m, "z1"],
                           s=14, alpha=0.7, color=color,
                           label=f"{cls} ({int(m.sum())})",
                           edgecolor="none")
        ax.set_title(f"{lig} (n={len(sub)})", fontsize=11)
        ax.legend(fontsize=8, frameon=False, loc="best")
    for k in range(n_lig, nrow*ncol):
        axes[k // ncol, k % ncol].axis("off")
    fig.suptitle("(C) Ligand-type latent footprints, coloured by Kincore DFG state",
                 fontsize=14, y=1.005)
    fig.tight_layout()
    fig.savefig(fig_dir / "ligand_type_latent_facets.png")
    plt.close(fig)

    print(f"\nAll outputs under {args.out}/")


if __name__ == "__main__":
    main()
