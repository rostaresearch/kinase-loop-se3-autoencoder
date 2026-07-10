"""Compare Cα-Cα vs min heavy-atom distances as features for predicting
the v9 activation-loop latent.

Two feature matrices are built over the SAME (n_chains × n_pairs) layout
using the SAME conserved non-loop residue list:

  X_ca   : Cα-only Euclidean distance, exactly the kinase pipeline's
           definition.
  X_min  : minimum over all heavy-atom pairs between residues i and j.

We then:
  1. Measure per-feature Pearson r between the two definitions across
     all chains (do the two metrics encode the same geometry?).
  2. Train identical LightGBM models on each.  Compare test R² for
     z0, z1, and combined.
  3. Compare top-20 features and per-residue importance between the
     two distance definitions.
  4. Save a side-by-side feature table.

Outputs (under --out):
  ca_vs_minatom_feature_correlation.csv     per-feature r + delta-mean
  ca_vs_minatom_model_comparison.csv        R² table
  ca_vs_minatom_top_features.csv            top-20 each way + intersection
  figures/ca_vs_minatom_scatter.png         per-feature mean (Cα vs min)
  figures/ca_vs_minatom_r2_bars.png         R² comparison
  figures/ca_vs_minatom_top_residues.png    consensus residue importance
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11, "axes.labelsize": 11, "axes.titlesize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def norm_key(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.replace("_", "", regex=False)


def read_atom_dict(pdb_path: Path, chain: str
                   ) -> dict[int, dict[str, np.ndarray]]:
    """Return {resi: {atom_name: (x,y,z)}} for heavy atoms on the
    specified chain.  Skips altloc != 'A' / '' and hydrogens."""
    out: dict[int, dict[str, np.ndarray]] = {}
    if not pdb_path.exists():
        return out
    with pdb_path.open() as f:
        for line in f:
            if not line.startswith("ATOM") or line[21] != chain:
                continue
            element = line[76:78].strip()
            if element == "H":
                continue
            atom = line[12:16].strip()
            if atom.startswith("H") and not element:
                continue
            alt = line[16].strip()
            if alt not in ("", "A"):
                continue
            try:
                resi = int(line[22:26])
            except ValueError:
                continue
            out.setdefault(resi, {})[atom] = np.array(
                [float(line[30:38]), float(line[38:46]),
                 float(line[46:54])], dtype=np.float32)
    return out


def build_matrices(conserved_csv: Path, manifest_csv: Path,
                   full_pdb_dir: Path, ref_chain_key: str,
                   ape_resi_floor: int, min_pair_coverage: float):
    """Build BOTH X_ca and X_min over the same residue-pair list."""
    conserved = pd.read_csv(conserved_csv, keep_default_na=False)
    conserved["chain_key"] = norm_key(conserved["chain_key"])
    ref = conserved[conserved["chain_key"] == ref_chain_key.upper()].copy()
    ref["pdb_resi"] = ref["pdb_resi"].astype(int)
    ref = ref[ref["pdb_resi"] < ape_resi_floor]
    braf_resis = sorted(ref["pdb_resi"].unique().tolist())
    print(f"Reference conserved non-loop BRAF residues: {len(braf_resis)}")

    pair_list = [(braf_resis[i], braf_resis[j])
                 for i in range(len(braf_resis))
                 for j in range(i + 1, len(braf_resis))]
    print(f"Initial pairs: {len(pair_list)}")

    manifest = pd.read_csv(manifest_csv, keep_default_na=False)
    manifest["chain_key"] = manifest["chain_key"].astype(str).str.upper()

    chain_maps = (
        conserved.groupby("chain_key")
        .apply(lambda g: dict(zip(g["braf_resi"].astype(int),
                                  g["pdb_resi"].astype(int))))
    ).to_dict()

    n = len(manifest)
    m = len(pair_list)
    X_ca = np.full((n, m), np.nan, dtype=np.float32)
    X_min = np.full((n, m), np.nan, dtype=np.float32)
    for ii in range(n):
        if ii % 500 == 0:
            print(f"  chain {ii}/{n}")
        row = manifest.iloc[ii]
        key = row["chain_key"]
        cmap = chain_maps.get(key)
        if cmap is None:
            continue
        atoms = read_atom_dict(full_pdb_dir / f"{row['pdb']}.pdb",
                                row["chain"])
        for jj, (ri, rj) in enumerate(pair_list):
            pi = cmap.get(ri); pj = cmap.get(rj)
            if pi is None or pj is None:
                continue
            ai = atoms.get(pi); aj = atoms.get(pj)
            if not ai or not aj:
                continue
            # Cα-Cα
            ca_i = ai.get("CA"); ca_j = aj.get("CA")
            if ca_i is not None and ca_j is not None:
                X_ca[ii, jj] = float(np.linalg.norm(ca_i - ca_j))
            # Min heavy-atom
            ci = np.array(list(ai.values()))
            cj = np.array(list(aj.values()))
            d = np.linalg.norm(ci[:, None, :] - cj[None, :, :], axis=-1)
            X_min[ii, jj] = float(d.min())
    # Coverage filter applied to both with the SAME mask (union of
    # missing-from-either) so we are comparing like for like.
    cov_ca = (~np.isnan(X_ca)).mean(axis=0)
    cov_min = (~np.isnan(X_min)).mean(axis=0)
    keep = (cov_ca >= min_pair_coverage) & (cov_min >= min_pair_coverage)
    print(f"Pairs with ≥{min_pair_coverage*100:.0f}% coverage in BOTH: "
          f"{keep.sum()}/{len(pair_list)}")
    X_ca = X_ca[:, keep]
    X_min = X_min[:, keep]
    pair_list = [p for p, k in zip(pair_list, keep) if k]
    # mean-impute
    for X in (X_ca, X_min):
        col_mean = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_mean, inds[1])
    return X_ca, X_min, manifest, pair_list


def per_feature_correlation(X_ca, X_min, pair_list, out: Path):
    rs = np.zeros(X_ca.shape[1])
    diffs = np.zeros(X_ca.shape[1])
    means_ca = np.zeros(X_ca.shape[1])
    means_min = np.zeros(X_ca.shape[1])
    for j in range(X_ca.shape[1]):
        a, b = X_ca[:, j], X_min[:, j]
        rs[j], _ = pearsonr(a, b)
        diffs[j] = (a - b).mean()
        means_ca[j] = a.mean()
        means_min[j] = b.mean()
    df = pd.DataFrame({
        "feature_idx": np.arange(X_ca.shape[1]),
        "resi_i": [p[0] for p in pair_list],
        "resi_j": [p[1] for p in pair_list],
        "pearson_r":  rs,
        "mean_ca":    means_ca,
        "mean_min":   means_min,
        "delta_mean": diffs,
    })
    df.to_csv(out / "ca_vs_minatom_feature_correlation.csv", index=False)
    print(f"\nper-feature r: median={np.median(rs):.3f}, "
          f"<0.5 count={int((rs<0.5).sum())}, "
          f"<0.7 count={int((rs<0.7).sum())}, "
          f">=0.95 count={int((rs>=0.95).sum())}")
    return df


def train_lgbm_both(X_ca, X_min, latent_csv: Path, manifest: pd.DataFrame,
                    seed: int, out: Path, conserved_csv: Path):
    """Train identical LightGBM on each feature matrix; report R²."""
    import lightgbm as lgb
    from sklearn.metrics import r2_score
    # Drop chains with no FoldMason mapping (same logic as
    # predict_v9_lgbm_shap.py)
    cons = pd.read_csv(conserved_csv, keep_default_na=False)
    cons["chain_key"] = norm_key(cons["chain_key"])
    mapped = set(cons["chain_key"].unique())
    manifest_keys = manifest["chain_key"].astype(str).str.upper().values
    has_mapping = np.array([k in mapped for k in manifest_keys])
    X_ca = X_ca[has_mapping]
    X_min = X_min[has_mapping]
    manifest = manifest[has_mapping].reset_index(drop=True)
    print(f"After dropping no-mapping chains: {X_ca.shape}")
    # Target alignment
    latent = pd.read_csv(latent_csv, keep_default_na=False)
    latent["chain_key"] = latent["chain_key"].astype(str).str.upper()
    latent = latent.set_index("chain_key")
    keys = manifest["chain_key"].tolist()
    keep_mask = np.array([k in latent.index for k in keys])
    X_ca = X_ca[keep_mask]; X_min = X_min[keep_mask]
    keys = [k for k in keys if k in latent.index]
    Y = latent.loc[keys, ["z0", "z1"]].to_numpy(dtype=np.float32)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Y))
    n_test = int(len(Y) * 0.1)
    test = perm[:n_test]; train = perm[n_test:]

    rows = []
    importance = {"ca": {}, "min": {}}
    for name, X in (("ca", X_ca), ("min", X_min)):
        for j, t in enumerate(["z0", "z1"]):
            m = lgb.LGBMRegressor(
                n_estimators=400, num_leaves=31, learning_rate=0.05,
                min_data_in_leaf=20, feature_fraction=0.9,
                bagging_fraction=0.9, bagging_freq=5,
                random_state=seed, n_jobs=-1, verbose=-1)
            m.fit(X[train], Y[train, j])
            p = m.predict(X[test])
            r2 = r2_score(Y[test, j], p)
            rows.append({"feature_set": name, "target": t, "r2_test": r2})
            print(f"  {name:3s} target {t}: R²_test = {r2:.4f}")
            importance[name][t] = m.booster_.feature_importance(
                importance_type="gain")
    pd.DataFrame(rows).to_csv(
        out / "ca_vs_minatom_model_comparison.csv", index=False)
    return importance


def compare_top_features(importance, pair_list, out: Path):
    rows = []
    overlaps = {}
    for t in ("z0", "z1"):
        ca = importance["ca"][t]
        mi = importance["min"][t]
        top_ca = np.argsort(-ca)[:20]
        top_mi = np.argsort(-mi)[:20]
        rank_ca = pd.Series(ca).rank(method="average", ascending=False)
        rank_mi = pd.Series(mi).rank(method="average", ascending=False)
        for idx in sorted(set(top_ca) | set(top_mi)):
            rows.append({
                "target": t, "feature_idx": idx,
                "resi_i": pair_list[idx][0],
                "resi_j": pair_list[idx][1],
                "ca_gain":  float(ca[idx]),
                "min_gain": float(mi[idx]),
                "rank_ca":  int(rank_ca.iloc[idx]),
                "rank_min": int(rank_mi.iloc[idx]),
                "in_top20_ca":  idx in set(top_ca),
                "in_top20_min": idx in set(top_mi),
            })
        n_overlap = len(set(top_ca) & set(top_mi))
        overlaps[t] = n_overlap
        print(f"  {t}: top-20 overlap = {n_overlap}/20 "
              f"({n_overlap/20*100:.0f}%)")
    pd.DataFrame(rows).to_csv(out / "ca_vs_minatom_top_features.csv",
                              index=False)
    return overlaps


def plot_summary(corr_df: pd.DataFrame, overlaps: dict,
                 importance: dict, pair_list, out: Path,
                 model_df: pd.DataFrame):
    # Plot 1: per-feature mean(Cα) vs mean(min-atom) scatter
    fig, ax = plt.subplots(figsize=(6, 5.5))
    sc = ax.scatter(corr_df["mean_ca"], corr_df["mean_min"], s=8,
                    c=corr_df["pearson_r"], cmap="viridis", alpha=0.6,
                    vmin=0, vmax=1)
    lim = max(corr_df["mean_ca"].max(), corr_df["mean_min"].max())
    ax.plot([0, lim], [0, lim], "k--", lw=0.7, alpha=0.5)
    ax.set_xlabel("Mean Cα–Cα distance (Å)")
    ax.set_ylabel("Mean min heavy-atom distance (Å)")
    ax.set_title("Per-feature: Cα–Cα vs min heavy-atom\n"
                  "(colour = per-feature Pearson r across chains)")
    fig.colorbar(sc, label="Pearson r")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"ca_vs_minatom_scatter.{ext}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: R² bars
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(2)
    width = 0.35
    ca = model_df[model_df["feature_set"] == "ca"]["r2_test"].values
    mn = model_df[model_df["feature_set"] == "min"]["r2_test"].values
    ax.bar(x - width / 2, ca, width, label="Cα–Cα", color="#1976D2")
    ax.bar(x + width / 2, mn, width, label="Min heavy-atom",
            color="#E65100")
    for i in range(2):
        ax.text(x[i] - width / 2, ca[i] + 0.005, f"{ca[i]:.3f}",
                ha="center", fontsize=9)
        ax.text(x[i] + width / 2, mn[i] + 0.005, f"{mn[i]:.3f}",
                ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(["z0", "z1"])
    ax.set_ylabel("LightGBM test R²")
    ax.set_title("Predicting v9 latent: Cα-Cα vs min heavy-atom features")
    ax.set_ylim(0, max(ca.max(), mn.max()) * 1.1)
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"ca_vs_minatom_r2_bars.{ext}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Plot 3: per-residue importance, both definitions
    def per_residue(imp, target):
        out_d = defaultdict(float)
        for j, (ri, rj) in enumerate(pair_list):
            r = pd.Series(imp[target]).rank(method="average",
                                              ascending=False).iloc[j]
            if r <= 50:
                out_d[ri] += 1 / r
                out_d[rj] += 1 / r
        return out_d
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for ax, t in zip(axes, ("z0", "z1")):
        ca = per_residue(importance["ca"], t)
        mn = per_residue(importance["min"], t)
        all_r = sorted(set(ca) | set(mn))
        x = np.array(all_r)
        y_ca = np.array([ca.get(r, 0) for r in all_r])
        y_mn = np.array([mn.get(r, 0) for r in all_r])
        ax.bar(x - 0.4, y_ca, width=0.8, color="#1976D2", alpha=0.85,
                label="Cα–Cα")
        ax.bar(x + 0.4, y_mn, width=0.8, color="#E65100", alpha=0.85,
                label="Min heavy-atom")
        ax.set_ylabel(f"Residue importance ({t})")
        ax.legend(frameon=False, loc="upper left")
    axes[1].set_xlabel("BRAF residue (Kincore numbering)")
    fig.suptitle("Per-residue importance (sum 1/rank over top-50 pairs)",
                  y=1.0)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"ca_vs_minatom_top_residues.{ext}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conserved-csv", required=True, type=Path)
    ap.add_argument("--manifest-csv", required=True, type=Path)
    ap.add_argument("--full-pdb-dir", required=True, type=Path)
    ap.add_argument("--latent-csv", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--ref-chain-key", default="6UANC")
    ap.add_argument("--ape-resi-floor", type=int, default=624)
    ap.add_argument("--min-pair-coverage", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=25)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out / "figures"; fig_dir.mkdir(exist_ok=True)

    print("=== Building feature matrices (Cα-Cα and min-atom) ===")
    X_ca, X_min, manifest, pair_list = build_matrices(
        args.conserved_csv, args.manifest_csv, args.full_pdb_dir,
        args.ref_chain_key, args.ape_resi_floor, args.min_pair_coverage)

    # Save matrices for downstream reuse
    np.savez_compressed(args.out / "matrices.npz",
                         X_ca=X_ca, X_min=X_min,
                         chain_keys=np.array(manifest["chain_key"]),
                         pair_list=np.array(pair_list, dtype=np.int32))
    print(f"\nSaved matrices.npz "
          f"(X_ca {X_ca.shape}, X_min {X_min.shape})")

    print("\n=== Per-feature correlation ===")
    corr_df = per_feature_correlation(X_ca, X_min, pair_list, args.out)

    print("\n=== Training LightGBM on each feature set ===")
    importance = train_lgbm_both(X_ca, X_min, args.latent_csv, manifest,
                                  args.seed, args.out, args.conserved_csv)
    model_df = pd.read_csv(args.out / "ca_vs_minatom_model_comparison.csv")

    print("\n=== Top-20 feature overlap ===")
    overlaps = compare_top_features(importance, pair_list, args.out)

    print("\n=== Plotting ===")
    plot_summary(corr_df, overlaps, importance, pair_list, fig_dir,
                 model_df)
    print("\nDone.  Outputs in", args.out)


if __name__ == "__main__":
    main()
