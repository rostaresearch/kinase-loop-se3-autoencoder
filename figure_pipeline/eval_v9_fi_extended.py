"""Extended feature-importance evaluation for the v9 LightGBM model.

Goes beyond the already-computed LightGBM gain + SHAP analysis:

  - **Permutation importance**: model-agnostic, computed by shuffling each
    feature on the held-out set and measuring the R² drop. Complementary
    to gain (which is internal-to-LightGBM) and SHAP.
  - **Ridge linear-coefficient importance**: |β| from a separately trained
    ridge regression. Lets us see whether the linear model agrees on
    which residues matter.
  - **Random Forest impurity importance**: a third tree-based importance.
  - **Cross-method ranking agreement**: Spearman ρ between every pair of
    methods, plotted as a heatmap.
  - **Class-stratified distance distributions**: for the top SHAP features,
    plot the actual Cα-Cα distance distribution coloured by Kincore DFG
    spatial class. Visually confirms the FI ranking picks up real
    DFG-in vs DFG-out signal.
  - **Quantitative FI quality**: train a simple linear logistic-regression
    classifier on the top-N FI features (by gain, SHAP, permutation, ridge),
    predicting Kincore DFG spatial. Reports test-set balanced accuracy
    as a function of N. If the FI ranking is meaningful, the top-N curves
    should rise faster than a random-feature baseline.

Inputs are the same as ``predict_v9_lgbm_shap.py`` plus the latent CSV
with Kincore labels and the v9 manifest.

Outputs (under ``--out``):
  - extended_fi_table.csv        — one row per feature, all 4 importance scores
  - ranking_agreement_heatmap.png — Spearman ρ between methods (4x4)
  - top_feature_distance_distributions_z0.png  / _z1.png
  - top_feature_classifier_curve.png   — DFG-class accuracy vs N features
  - top_feature_classifier_table.csv
  - per_residue_importance_combined.png — overlay of 4 methods on residue axis
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
    "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 14,
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


def build_distance_matrix(conserved_csv, manifest_csv, full_pdb_dir,
                          ref_chain_key, ape_resi_floor, min_pair_coverage):
    conserved = pd.read_csv(conserved_csv, keep_default_na=False)
    conserved["chain_key"] = norm_key(conserved["chain_key"])
    ref = conserved[conserved["chain_key"] == ref_chain_key.upper()].copy()
    ref["pdb_resi"] = ref["pdb_resi"].astype(int)
    ref = ref[ref["pdb_resi"] < ape_resi_floor]
    braf_resis = sorted(ref["pdb_resi"].unique().tolist())
    pair_list = [(braf_resis[i], braf_resis[j])
                 for i in range(len(braf_resis))
                 for j in range(i + 1, len(braf_resis))]

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
        cmap = chain_maps.get(row["chain_key"])
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
    X = X[:, keep]
    pair_list = [p for p, k in zip(pair_list, keep) if k]
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])
    return X, manifest, pair_list


def per_axis_r2(y_true, y_pred):
    r2s = []
    for j in range(y_true.shape[1]):
        yt = y_true[:, j]; yp = y_pred[:, j]
        ss_res = ((yt - yp) ** 2).sum()
        ss_tot = ((yt - yt.mean()) ** 2).sum()
        r2s.append(1 - ss_res / ss_tot if ss_tot > 0 else np.nan)
    return r2s


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

    # Build features.
    print("Building distance matrix")
    X, manifest, pairs = build_distance_matrix(
        args.conserved_csv, args.manifest_csv, args.full_pdb_dir,
        args.ref_chain_key, args.ape_resi_floor, args.min_pair_coverage,
    )
    feature_names = [f"d_{a}_{b}" for a, b in pairs]

    # BUG-FIX 2026-05-27: drop chains with no FoldMason mapping
    conserved_df = pd.read_csv(args.conserved_csv, keep_default_na=False)
    conserved_df["chain_key"] = norm_key(conserved_df["chain_key"])
    mapped_keys = set(conserved_df["chain_key"].unique())
    has_mapping = manifest["chain_key"].str.upper().isin(mapped_keys).values
    n_drop = int((~has_mapping).sum())
    if n_drop:
        print(f"Dropping {n_drop} chains with no FoldMason mapping")
    X = X[has_mapping]
    manifest = manifest[has_mapping].reset_index(drop=True)

    n_features = X.shape[1]
    print(f"Cleaned X={X.shape}")

    # Targets.
    latent = pd.read_csv(args.latent_csv, keep_default_na=False)
    latent["chain_key"] = latent["chain_key"].astype(str).str.upper()
    latent = latent.set_index("chain_key")
    keys = manifest["chain_key"].tolist()
    mask = np.array([k in latent.index for k in keys])
    X = X[mask]
    keys = [k for k in keys if k in latent.index]
    Y = latent.loc[keys, ["z0", "z1"]].to_numpy(dtype=np.float32)
    dfg_labels = latent.loc[keys, "dfg_spatial"].fillna("None").astype(str).values

    # Split.
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(X))
    n_test = int(len(X) * 0.1)
    test = perm[:n_test]; train = perm[n_test:]
    Xtr, Xte = X[train], X[test]
    Ytr, Yte = Y[train], Y[test]

    # Standardise (for ridge).
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    Xtr_s = sc.transform(Xtr); Xte_s = sc.transform(Xte)

    # ---------- 1. LightGBM ----------
    print("\nTraining LightGBM (z0, z1)")
    import lightgbm as lgb
    lgb_models = {}
    lgb_preds = np.zeros_like(Yte)
    for j, name in enumerate(["z0", "z1"]):
        m = lgb.LGBMRegressor(
            num_leaves=31, n_estimators=400, learning_rate=0.05,
            feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=5,
            min_data_in_leaf=20, verbose=-1, random_state=args.seed)
        m.fit(Xtr, Ytr[:, j])
        lgb_models[name] = m
        lgb_preds[:, j] = m.predict(Xte)
    lgb_r2 = per_axis_r2(Yte, lgb_preds)
    print(f"  LGBM z0 R²={lgb_r2[0]:.3f}, z1 R²={lgb_r2[1]:.3f}")

    gain = {name: m.booster_.feature_importance(importance_type="gain")
            for name, m in lgb_models.items()}

    # ---------- 2. SHAP ----------
    print("\nComputing SHAP on test set")
    import shap
    shap_abs = {}
    for name, m in lgb_models.items():
        sv = shap.TreeExplainer(m).shap_values(Xte)
        if isinstance(sv, list): sv = sv[0]
        shap_abs[name] = np.abs(sv).mean(axis=0)

    # ---------- 3. Permutation importance on LightGBM ----------
    print("\nComputing permutation importance on LightGBM test set")
    from sklearn.inspection import permutation_importance
    perm_imp = {}
    for j, name in enumerate(["z0", "z1"]):
        result = permutation_importance(
            lgb_models[name], Xte, Yte[:, j],
            n_repeats=5, random_state=args.seed, n_jobs=-1,
            scoring="r2",
        )
        perm_imp[name] = result.importances_mean
        print(f"  {name}: top 5 perm-imp features:",
              [f"{pairs[i][0]}-{pairs[i][1]}"
               for i in np.argsort(-result.importances_mean)[:5]])

    # ---------- 4. Ridge linear-coefficient importance ----------
    print("\nTraining ridge for coefficient importance")
    from sklearn.linear_model import Ridge
    ridge_models = {}
    ridge_preds = np.zeros_like(Yte)
    ridge_abs_coef = {}
    for j, name in enumerate(["z0", "z1"]):
        r = Ridge(alpha=3000.0, random_state=args.seed)
        r.fit(Xtr_s, Ytr[:, j])
        ridge_models[name] = r
        ridge_preds[:, j] = r.predict(Xte_s)
        ridge_abs_coef[name] = np.abs(r.coef_)
    ridge_r2 = per_axis_r2(Yte, ridge_preds)
    print(f"  Ridge z0 R²={ridge_r2[0]:.3f}, z1 R²={ridge_r2[1]:.3f}")

    # ---------- 5. Random Forest impurity importance ----------
    print("\nTraining Random Forest")
    from sklearn.ensemble import RandomForestRegressor
    rf_imp = {}
    rf_preds = np.zeros_like(Yte)
    for j, name in enumerate(["z0", "z1"]):
        r = RandomForestRegressor(
            n_estimators=300, min_samples_leaf=4,
            n_jobs=-1, random_state=args.seed)
        r.fit(Xtr, Ytr[:, j])
        rf_imp[name] = r.feature_importances_
        rf_preds[:, j] = r.predict(Xte)
    rf_r2 = per_axis_r2(Yte, rf_preds)
    print(f"  RF z0 R²={rf_r2[0]:.3f}, z1 R²={rf_r2[1]:.3f}")

    # ---------- Master FI table ----------
    rows = []
    for j, (a, b) in enumerate(pairs):
        for tg in ("z0", "z1"):
            rows.append({
                "target": tg,
                "feature": feature_names[j],
                "resi_i": a, "resi_j": b,
                "lgbm_gain": float(gain[tg][j]),
                "lgbm_shap_meanabs": float(shap_abs[tg][j]),
                "lgbm_permutation": float(perm_imp[tg][j]),
                "ridge_abs_coef": float(ridge_abs_coef[tg][j]),
                "rf_impurity": float(rf_imp[tg][j]),
            })
    fi_df = pd.DataFrame(rows)
    fi_df.to_csv(args.out / "extended_fi_table.csv", index=False)

    # ---------- Cross-method ranking agreement (Spearman) ----------
    from scipy.stats import spearmanr
    methods = ["lgbm_gain", "lgbm_shap_meanabs", "lgbm_permutation",
               "ridge_abs_coef", "rf_impurity"]
    for tg in ("z0", "z1"):
        sub = fi_df[fi_df["target"] == tg]
        n_m = len(methods)
        rho = np.zeros((n_m, n_m))
        for i, mi in enumerate(methods):
            for j, mj in enumerate(methods):
                rho[i, j] = spearmanr(sub[mi], sub[mj]).correlation
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        im = ax.imshow(rho, vmin=-0.2, vmax=1.0, cmap="RdBu_r")
        ax.set_xticks(range(n_m)); ax.set_yticks(range(n_m))
        ax.set_xticklabels([m.replace("_", "\n") for m in methods],
                           rotation=30, ha="right")
        ax.set_yticklabels([m.replace("_", "\n") for m in methods])
        for i in range(n_m):
            for j in range(n_m):
                ax.text(j, i, f"{rho[i,j]:.2f}", ha="center", va="center",
                        color="black" if abs(rho[i,j]) < 0.6 else "white",
                        fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.04, label="Spearman ρ")
        ax.set_title(f"Feature-importance ranking agreement — {tg}")
        fig.tight_layout()
        fig.savefig(args.out / f"ranking_agreement_{tg}.png")
        plt.close(fig)

    # ---------- Per-residue importance overlay (4 methods) ----------
    all_resis = sorted({r for p in pairs for r in p})
    res_idx = {r: i for i, r in enumerate(all_resis)}
    n_res = len(all_resis)

    def aggregate(method_per_target):
        """Sum the per-feature importance to per-residue, summed over targets."""
        agg = np.zeros(n_res)
        for tg in ("z0", "z1"):
            for j, (a, b) in enumerate(pairs):
                v = method_per_target[tg][j]
                agg[res_idx[a]] += v
                agg[res_idx[b]] += v
        return agg / agg.max() if agg.max() > 0 else agg

    methods_dict = {
        "LightGBM gain": gain,
        "LightGBM SHAP": shap_abs,
        "Permutation imp.": perm_imp,
        "Ridge |coef|": ridge_abs_coef,
        "RF impurity": rf_imp,
    }
    fig, ax = plt.subplots(figsize=(15, 6))
    colors = ["#315f8e", "#c0504d", "#6ACC64", "#9c6dc3", "#e88a00"]
    width = 0.18
    for k, (name, mpt) in enumerate(methods_dict.items()):
        agg = aggregate(mpt)
        ax.bar(np.array(all_resis) + (k - 2) * width, agg, width=width,
               label=name, color=colors[k % len(colors)])
    ax.set_xlabel("BRAF residue number")
    ax.set_ylabel("Normalised per-residue importance (Σ over targets)")
    ax.set_title("Per-residue feature importance across methods (normalised within method)")
    ax.legend(loc="upper left", frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(args.out / "per_residue_importance_combined.png")
    plt.close(fig)

    # ---------- Top-feature class-stratified distance distributions ----------
    for tg in ("z0", "z1"):
        sub = fi_df[fi_df["target"] == tg].sort_values(
            "lgbm_shap_meanabs", ascending=False)
        top_idxs = [feature_names.index(f) for f in sub["feature"].head(9)]
        fig, axes = plt.subplots(3, 3, figsize=(12, 9))
        for k, fi in enumerate(top_idxs):
            ax = axes[k // 3, k % 3]
            for cls, color in [("DFGin", "#315f8e"), ("DFGout", "#c0504d"),
                               ("DFGinter", "#6ACC64")]:
                m = (dfg_labels == cls)
                if m.sum() == 0: continue
                ax.hist(X[m, fi], bins=40, alpha=0.55, density=True,
                        label=f"{cls} (n={m.sum()})", color=color)
            ri, rj = pairs[fi]
            ax.set_xlabel(f"d({ri},{rj}) [Å]")
            ax.set_ylabel("density")
            ax.set_title(f"#{k+1} {ri}-{rj}", fontsize=11)
            if k == 0:
                ax.legend(fontsize=8, frameon=False)
        fig.suptitle(f"Distance distributions for top-9 SHAP features predicting {tg}, "
                     f"stratified by DFG spatial class", fontsize=13)
        fig.tight_layout()
        fig.savefig(args.out / f"top_feature_distance_distributions_{tg}.png")
        plt.close(fig)

    # ---------- Quantitative FI quality: top-N → DFG classifier ----------
    print("\nTop-N feature DFG-class classifier")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    valid = (dfg_labels != "None") & (dfg_labels != "")
    Xv = X[valid]; yv = dfg_labels[valid]
    # split
    tr_v = np.intersect1d(np.where(valid)[0], train)
    te_v = np.intersect1d(np.where(valid)[0], test)
    rows = []
    Ns = [3, 5, 10, 20, 30, 50, 100, 200, 400, n_features]
    # combined importance per method (sum over z0+z1)
    combined = {
        "lgbm_gain": gain["z0"] + gain["z1"],
        "lgbm_shap_meanabs": shap_abs["z0"] + shap_abs["z1"],
        "lgbm_permutation": perm_imp["z0"] + perm_imp["z1"],
        "ridge_abs_coef": ridge_abs_coef["z0"] + ridge_abs_coef["z1"],
        "rf_impurity": rf_imp["z0"] + rf_imp["z1"],
    }
    # random baseline (mean over 5 random selections)
    # NOTE: the conserved-distance features are raw Cα–Cα distances (tens of Å),
    # so lbfgs LogisticRegression on the un-standardised matrix fails to converge
    # in 1000 iters (esp. at large N). Standardise per fit (StandardScaler) — this
    # both removes the ConvergenceWarning and makes each fit converge far faster.
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    def _dfg_probe():
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, n_jobs=-1),
        )

    rng2 = np.random.default_rng(args.seed)
    for N in Ns:
        for method_name, imp in combined.items():
            top = np.argsort(-imp)[:N]
            clf = _dfg_probe()
            clf.fit(X[tr_v][:, top], dfg_labels[tr_v])
            yhat = clf.predict(X[te_v][:, top])
            bal = balanced_accuracy_score(dfg_labels[te_v], yhat)
            rows.append({"method": method_name, "N": N, "bal_acc": float(bal)})
        # random baseline
        random_accs = []
        for _ in range(5):
            top = rng2.permutation(n_features)[:N]
            clf = _dfg_probe()
            clf.fit(X[tr_v][:, top], dfg_labels[tr_v])
            yhat = clf.predict(X[te_v][:, top])
            random_accs.append(balanced_accuracy_score(dfg_labels[te_v], yhat))
        rows.append({"method": "random", "N": N,
                     "bal_acc": float(np.mean(random_accs))})

    cls_df = pd.DataFrame(rows)
    cls_df.to_csv(args.out / "top_feature_classifier_table.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    method_colors = {"lgbm_gain": "#315f8e", "lgbm_shap_meanabs": "#c0504d",
                     "lgbm_permutation": "#6ACC64",
                     "ridge_abs_coef": "#9c6dc3", "rf_impurity": "#e88a00",
                     "random": "grey"}
    for method_name in method_colors:
        sub = cls_df[cls_df["method"] == method_name].sort_values("N")
        ax.plot(sub["N"], sub["bal_acc"], marker="o",
                color=method_colors[method_name],
                label=method_name, lw=2 if method_name != "random" else 1.5,
                ls="-" if method_name != "random" else "--",
                markersize=5)
    ax.set_xscale("log")
    ax.set_xlabel("Number of top features used (log scale)")
    ax.set_ylabel("DFG-class balanced accuracy (held-out test)")
    ax.set_title("Quantitative FI quality: predicting DFG spatial from top-N features\n"
                 "(direct logistic regression, no AE; the steeper the rise, the better the FI ranking)")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out / "top_feature_classifier_curve.png")
    plt.close(fig)

    # ---------- Headline numbers ----------
    print("\n========= SUMMARY =========")
    print(f"LightGBM z0 R² = {lgb_r2[0]:.3f}, z1 R² = {lgb_r2[1]:.3f}")
    print(f"Ridge    z0 R² = {ridge_r2[0]:.3f}, z1 R² = {ridge_r2[1]:.3f}")
    print(f"RF       z0 R² = {rf_r2[0]:.3f}, z1 R² = {rf_r2[1]:.3f}")
    print(f"\nFI tables and figures under {args.out}/")


if __name__ == "__main__":
    main()
