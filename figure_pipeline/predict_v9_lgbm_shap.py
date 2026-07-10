"""Train LightGBM on non-loop conserved Cα-Cα distances → predict v9 AE latent.

Then compute feature importance (gain + split) and SHAP values, and
produce figures suitable for the manuscript HTML report.

Output figures (all under ``--out/figures/``):
  - lgbm_predicted_vs_actual.png    — scatter of predicted vs actual z0, z1
  - lgbm_top_features_z0.png        — top 20 features by gain for z0
  - lgbm_top_features_z1.png        — top 20 features by gain for z1
  - lgbm_shap_summary_z0.png        — SHAP beeswarm for z0 (top 15)
  - lgbm_shap_summary_z1.png        — SHAP beeswarm for z1
  - lgbm_shap_bar_z0.png            — SHAP mean |abs| bar for top 20, z0
  - lgbm_shap_bar_z1.png            — same for z1
  - lgbm_residue_importance.png     — per-BRAF-residue aggregated SHAP (sum |abs|)
  - lgbm_feature_pairs_top.csv      — top-200 features by combined |SHAP|

Plus model_comparison.csv with full cross-validated R² breakdown.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 13, "axes.labelsize": 14, "axes.titlesize": 15,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def norm_key(s):
    return s.astype(str).str.upper().str.replace("_", "", regex=False)


def read_ca_map(pdb_path: Path, chain: str) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    if not pdb_path.exists():
        return out
    with pdb_path.open() as f:
        for line in f:
            if not line.startswith("ATOM") or line[21] != chain:
                continue
            if line[12:16].strip() != "CA":
                continue
            if line[16].strip() not in {"", "A"}:
                continue
            try:
                resi = int(line[22:26])
            except ValueError:
                continue
            out[resi] = np.array(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                dtype=np.float32,
            )
    return out


def build_distance_matrix(conserved_csv: Path, manifest_csv: Path,
                          full_pdb_dir: Path, ref_chain_key: str,
                          ape_resi_floor: int, min_pair_coverage: float):
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
    X = np.full((n, len(pair_list)), np.nan, dtype=np.float32)
    for ii in range(n):
        if ii % 500 == 0:
            print(f"  loading chain {ii}/{n}")
        row = manifest.iloc[ii]
        key = row["chain_key"]
        cmap = chain_maps.get(key)
        if cmap is None:
            continue
        cas = read_ca_map(full_pdb_dir / f"{row['pdb']}.pdb", row["chain"])
        for jj, (ri, rj) in enumerate(pair_list):
            pi = cmap.get(ri); pj = cmap.get(rj)
            if pi is None or pj is None:
                continue
            ci = cas.get(pi); cj = cas.get(pj)
            if ci is None or cj is None:
                continue
            X[ii, jj] = float(np.linalg.norm(ci - cj))
    coverage = (~np.isnan(X)).mean(axis=0)
    keep = coverage >= min_pair_coverage
    print(f"Pairs with ≥{min_pair_coverage*100:.0f}% coverage: "
          f"{keep.sum()}/{len(pair_list)}")
    X = X[:, keep]
    pair_list = [p for p, k in zip(pair_list, keep) if k]
    # Per-chain imputation fraction (BEFORE filling), so callers can drop
    # chains whose conserved-distance features are mostly mean-imputed.
    # Such chains have a FoldMason table entry but their mapped residues
    # don't actually cover the conserved non-loop pairs; left in, they
    # collapse to the centroid prediction and form a horizontal band in
    # the predicted-vs-actual scatter (residual form of the v9 "stripe").
    frac_imputed = np.isnan(X).mean(axis=1)
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])
    return X, manifest, pair_list, frac_imputed


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
    ap.add_argument("--max-imputed-frac", type=float, default=0.5,
                    help="Drop chains whose per-chain fraction of "
                         "mean-imputed conserved-distance features exceeds "
                         "this (centroid-collapse band fix). 0.5 = require "
                         "at least half the conserved pairs to be real.")
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--num-leaves", type=int, default=31)
    ap.add_argument("--seed", type=int, default=25)
    ap.add_argument("--n-top", type=int, default=20)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out / "figures"; fig_dir.mkdir(exist_ok=True)

    # Build features.
    X, manifest, pairs, frac_imputed = build_distance_matrix(
        args.conserved_csv, args.manifest_csv, args.full_pdb_dir,
        args.ref_chain_key, args.ape_resi_floor, args.min_pair_coverage,
    )
    feature_names = [f"d_{a}_{b}" for a, b in pairs]

    # BUG-FIX 2026-05-27 / completed 2026-06-18: drop chains whose feature
    # vector is dominated by column-mean imputation. Two cases:
    #  (1) no FoldMason mapping entry at all;
    #  (2) HAS an entry but its mapped residues don't actually cover the
    #      conserved non-loop pairs, so >50 % of features are mean-imputed.
    # Either way the chain collapses to the centroid prediction and forms a
    # horizontal band in predicted-vs-actual (residual form of the v9
    # "stripe"). Case (2) slipped past the original has-mapping-only filter.
    conserved_df = pd.read_csv(args.conserved_csv, keep_default_na=False)
    conserved_df["chain_key"] = norm_key(conserved_df["chain_key"])
    mapped_keys = set(conserved_df["chain_key"].unique())
    manifest_keys = manifest["chain_key"].astype(str).str.upper().values
    has_mapping = np.array([k in mapped_keys for k in manifest_keys])
    well_covered = frac_imputed <= args.max_imputed_frac
    keep_chain = has_mapping & well_covered
    print(f"Dropping {int((~has_mapping).sum())} chains with no FoldMason "
          f"mapping; {int((has_mapping & ~well_covered).sum())} more with "
          f">{args.max_imputed_frac*100:.0f}% imputed features "
          f"(centroid-collapse band).")
    X = X[keep_chain]
    manifest = manifest[keep_chain].reset_index(drop=True)
    print(f"Cleaned feature matrix: {X.shape}")

    # Targets.
    latent = pd.read_csv(args.latent_csv, keep_default_na=False)
    latent["chain_key"] = latent["chain_key"].astype(str).str.upper()
    latent = latent.set_index("chain_key")
    keys = manifest["chain_key"].tolist()
    keep_mask = np.array([k in latent.index for k in keys])
    X = X[keep_mask]
    keys = [k for k in keys if k in latent.index]
    Y = latent.loc[keys, ["z0", "z1"]].to_numpy(dtype=np.float32)
    print(f"X={X.shape}, Y={Y.shape}")

    # Split.
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(X))
    n_test = int(len(X) * 0.1)
    test = perm[:n_test]; train = perm[n_test:]
    X_train, X_test = X[train], X[test]
    Y_train, Y_test = Y[train], Y[test]

    # Train two LightGBM models, one per target.
    import lightgbm as lgb
    models = {}
    preds = np.zeros_like(Y_test)
    for j, name in enumerate(["z0", "z1"]):
        print(f"\nTraining LightGBM for {name}")
        m = lgb.LGBMRegressor(
            num_leaves=args.num_leaves,
            n_estimators=args.n_estimators,
            learning_rate=0.05,
            feature_fraction=0.7,
            bagging_fraction=0.7,
            bagging_freq=5,
            min_data_in_leaf=20,
            verbose=-1,
            random_state=args.seed,
        )
        m.fit(X_train, Y_train[:, j],
              feature_name=feature_names)
        models[name] = m
        preds[:, j] = m.predict(X_test)
        ss_res = ((Y_test[:, j] - preds[:, j]) ** 2).sum()
        ss_tot = ((Y_test[:, j] - Y_test[:, j].mean()) ** 2).sum()
        print(f"  {name} test R²: {1 - ss_res/ss_tot:.3f}")

    # Combined test R²
    ss_res = ((Y_test - preds) ** 2).sum()
    ss_tot = ((Y_test - Y_test.mean(axis=0)) ** 2).sum()
    combined_r2 = 1 - ss_res / ss_tot
    print(f"Combined R²: {combined_r2:.3f}")

    # ---------- Export test predictions (for external/MATLAB plotting) ----------
    pd.DataFrame({
        "actual_z0": Y_test[:, 0], "pred_z0": preds[:, 0],
        "actual_z1": Y_test[:, 1], "pred_z1": preds[:, 1],
    }).to_csv(args.out / "lgbm_test_predictions.csv", index=False)

    # ---------- Predicted vs actual scatter ----------
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for j, name in enumerate(["z0", "z1"]):
        axes[j].scatter(Y_test[:, j], preds[:, j], s=10, alpha=0.55,
                        color="#315f8e", edgecolor="none")
        mn = float(min(Y_test[:, j].min(), preds[:, j].min()))
        mx = float(max(Y_test[:, j].max(), preds[:, j].max()))
        axes[j].plot([mn, mx], [mn, mx], "k--", lw=1, alpha=0.6)
        ss_res = ((Y_test[:, j] - preds[:, j]) ** 2).sum()
        ss_tot = ((Y_test[:, j] - Y_test[:, j].mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
        axes[j].set_xlabel(f"Actual {name}")
        axes[j].set_ylabel(f"Predicted {name}")
        axes[j].set_title(f"{name}: R² = {r2:.3f}")
    fig.suptitle("LightGBM predicting v9.1 latent from non-loop conserved distances",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(fig_dir / "lgbm_predicted_vs_actual.png")
    plt.close(fig)

    # ---------- Built-in gain importance ----------
    rows = []
    for name, m in models.items():
        imps = m.booster_.feature_importance(importance_type="gain")
        idx = np.argsort(-imps)
        for rank in range(min(args.n_top, len(imps))):
            j = idx[rank]
            rows.append({
                "target": name, "rank": rank + 1,
                "feature": feature_names[j],
                "resi_i": pairs[j][0], "resi_j": pairs[j][1],
                "gain": float(imps[j]),
            })
    pd.DataFrame(rows).to_csv(args.out / "lgbm_gain_top.csv", index=False)

    for name, m in models.items():
        imps = m.booster_.feature_importance(importance_type="gain")
        idx = np.argsort(-imps)[:args.n_top]
        labels = [f"{pairs[j][0]}-{pairs[j][1]}" for j in idx]
        values = imps[idx]
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.barh(range(len(labels)), values[::-1], color="#315f8e")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels[::-1])
        ax.set_xlabel("Gain (LightGBM)")
        ax.set_title(f"Top {args.n_top} distance pairs by gain — predicting {name}")
        fig.tight_layout()
        fig.savefig(fig_dir / f"lgbm_top_features_{name}.png")
        plt.close(fig)

    # ---------- SHAP ----------
    import shap
    print("\nComputing SHAP values on test set")
    shap_values = {}
    for name, m in models.items():
        explainer = shap.TreeExplainer(m)
        sv = explainer.shap_values(X_test)
        # Some shap versions return list[2-d arr], handle both
        if isinstance(sv, list):
            sv = sv[0]
        shap_values[name] = sv
        print(f"  {name}: SHAP shape {sv.shape}")

    # SHAP top-feature bar (mean |abs|)
    shap_rows = []
    for name, sv in shap_values.items():
        mean_abs = np.abs(sv).mean(axis=0)
        idx = np.argsort(-mean_abs)
        for rank in range(min(args.n_top, len(mean_abs))):
            j = idx[rank]
            shap_rows.append({
                "target": name, "rank": rank + 1,
                "feature": feature_names[j],
                "resi_i": pairs[j][0], "resi_j": pairs[j][1],
                "mean_abs_shap": float(mean_abs[j]),
            })
    pd.DataFrame(shap_rows).to_csv(
        args.out / "lgbm_shap_top.csv", index=False)

    for name, sv in shap_values.items():
        mean_abs = np.abs(sv).mean(axis=0)
        idx = np.argsort(-mean_abs)[:args.n_top]
        labels = [f"{pairs[j][0]}-{pairs[j][1]}" for j in idx]
        values = mean_abs[idx]
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.barh(range(len(labels)), values[::-1], color="#c0504d")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels[::-1])
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"Top {args.n_top} SHAP features — predicting {name}")
        fig.tight_layout()
        fig.savefig(fig_dir / f"lgbm_shap_bar_{name}.png")
        plt.close(fig)

        # Beeswarm summary
        plt.figure(figsize=(8, 6))
        shap.summary_plot(sv, X_test, feature_names=feature_names,
                          max_display=15, show=False, plot_size=None)
        plt.title(f"SHAP summary for {name}")
        plt.tight_layout()
        plt.savefig(fig_dir / f"lgbm_shap_summary_{name}.png")
        plt.close()

    # ---------- Per-residue aggregated SHAP ----------
    # Each residue's importance = sum over all pairs it participates in
    all_resis = sorted({r for p in pairs for r in p})
    n_res = len(all_resis)
    resi_to_idx = {r: i for i, r in enumerate(all_resis)}
    agg = np.zeros((2, n_res), dtype=np.float64)
    for k, (name, sv) in enumerate(shap_values.items()):
        mean_abs = np.abs(sv).mean(axis=0)
        for j, (a, b) in enumerate(pairs):
            agg[k, resi_to_idx[a]] += mean_abs[j]
            agg[k, resi_to_idx[b]] += mean_abs[j]

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    for k, name in enumerate(["z0", "z1"]):
        axes[k].bar(all_resis, agg[k], color="#315f8e" if k == 0 else "#c0504d",
                    width=1.0)
        axes[k].set_ylabel(f"Σ |SHAP|  ({name})")
        axes[k].set_title(f"Per-BRAF-residue summed |SHAP| — predicting {name}")
    axes[1].set_xlabel("BRAF residue number")
    fig.tight_layout()
    fig.savefig(fig_dir / "lgbm_residue_importance.png")
    plt.close(fig)

    # Save residue importance table
    res_df = pd.DataFrame({
        "braf_resi": all_resis,
        "shap_z0": agg[0],
        "shap_z1": agg[1],
        "shap_combined": agg.sum(axis=0),
    })
    res_df.to_csv(args.out / "lgbm_residue_importance.csv", index=False)

    # Save top-200 distance pairs by combined |SHAP|
    combined_imp = np.zeros(len(pairs))
    for sv in shap_values.values():
        combined_imp += np.abs(sv).mean(axis=0)
    idx = np.argsort(-combined_imp)[:200]
    pd.DataFrame({
        "rank": range(1, len(idx) + 1),
        "resi_i": [pairs[i][0] for i in idx],
        "resi_j": [pairs[i][1] for i in idx],
        "shap_combined": combined_imp[idx],
    }).to_csv(args.out / "lgbm_feature_pairs_top.csv", index=False)

    # Summary CSV
    pd.DataFrame([{
        "model": "LightGBM",
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_features": int(X.shape[1]),
        "z0_test_r2": float(1 - ((Y_test[:,0]-preds[:,0])**2).sum() /
                            ((Y_test[:,0]-Y_test[:,0].mean())**2).sum()),
        "z1_test_r2": float(1 - ((Y_test[:,1]-preds[:,1])**2).sum() /
                            ((Y_test[:,1]-Y_test[:,1].mean())**2).sum()),
        "combined_r2": float(combined_r2),
    }]).to_csv(args.out / "lgbm_summary.csv", index=False)
    print(f"\nFigures and CSVs written under {args.out}/")


if __name__ == "__main__":
    main()
