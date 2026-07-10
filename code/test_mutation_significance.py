"""Statistical test: is the Δlatent of a mutation significant against WT spread?

For each (gene, mutation) pair where v9 has both WT and mutant chains:

  - Δ_obs = ‖mean(WT) − mean(mut)‖   (Euclidean in 2-D z0,z1)
  - σ_WT   = ((std(WT z0))² + (std(WT z1))²)^0.5
  - Mahalanobis(mut_centroid; WT distribution) — Δ in units of WT covariance
  - Permutation test p-value (10,000 shuffles of WT/mut labels)
  - Bootstrap 95 % CI on Δ_obs

Outputs:
  significance_summary.csv  one row per mutation
  significance_scatter.png  z0/z1 scatter with WT cloud + mutant centroid
                            + 95% confidence ellipse for each tested mutation
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
from scipy.stats import chi2

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def find_chains_for_mutation(df: pd.DataFrame, gene: str, mut: str) -> pd.DataFrame:
    sub = df[df["gene"] == gene]
    pat = re.compile(rf"\b{re.escape(mut)}\b", re.IGNORECASE)
    return sub[sub.apply(lambda r:
        bool(pat.search(str(r.get("title_mutation_hits", "")))
             or pat.search(str(r.get("seqadv_mutations", "")))
             or pat.search(str(r.get("remark_999_mutation_lines", "")))), axis=1)]


def mahalanobis_sq(p: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> float:
    """Squared Mahalanobis distance of point p from distribution N(mu, cov)."""
    diff = p - mu
    inv = np.linalg.pinv(cov)
    return float(diff @ inv @ diff)


def permutation_pvalue(wt: np.ndarray, mut: np.ndarray, n_perm: int = 10000,
                       seed: int = 25) -> tuple[float, float]:
    """Permutation test: are mean(wt) and mean(mut) further apart than chance?

    Returns (p_value, observed_distance).
    """
    rng = np.random.default_rng(seed)
    n_wt = len(wt); n_mut = len(mut)
    pooled = np.vstack([wt, mut])
    obs = float(np.linalg.norm(pooled[:n_wt].mean(0) - pooled[n_wt:].mean(0)))
    if n_wt + n_mut < 4:
        return 1.0, obs
    null = np.zeros(n_perm)
    for i in range(n_perm):
        idx = rng.permutation(n_wt + n_mut)
        a = pooled[idx[:n_wt]].mean(0)
        b = pooled[idx[n_wt:]].mean(0)
        null[i] = float(np.linalg.norm(a - b))
    p = float((null >= obs).sum() + 1) / (n_perm + 1)
    return p, obs


def bootstrap_delta(wt: np.ndarray, mut: np.ndarray, n_boot: int = 5000,
                     seed: int = 25) -> tuple[float, float, float]:
    """Bootstrap 95 % CI for mean(WT) - mean(mut) distance."""
    rng = np.random.default_rng(seed)
    n_wt, n_mut = len(wt), len(mut)
    if min(n_wt, n_mut) < 2:
        return float("nan"), float("nan"), float("nan")
    boots = np.zeros(n_boot)
    for i in range(n_boot):
        a = wt[rng.integers(0, n_wt, size=n_wt)].mean(0)
        b = mut[rng.integers(0, n_mut, size=n_mut)].mean(0)
        boots[i] = float(np.linalg.norm(a - b))
    return float(np.median(boots)), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def plot_one_mutation(ax, wt: np.ndarray, mut: np.ndarray,
                       gene: str, mutation: str, result: dict):
    # WT scatter
    ax.scatter(wt[:, 0], wt[:, 1], s=18, alpha=0.55, color="#4c78a8",
               label=f"WT (n={len(wt)})", edgecolor="none")
    # Mutant scatter
    ax.scatter(mut[:, 0], mut[:, 1], s=80, alpha=0.85, color="#c0504d",
               label=f"{mutation} (n={len(mut)})", edgecolor="black",
               linewidth=0.8)
    # WT 95% confidence ellipse
    mu = wt.mean(0); cov = np.cov(wt.T)
    chi2_95 = chi2.ppf(0.95, df=2)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]; vecs = vecs[:, order]
    theta = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
    w, h = 2 * np.sqrt(vals * chi2_95)
    ell = Ellipse(xy=mu, width=w, height=h, angle=theta,
                  edgecolor="#4c78a8", facecolor="none", lw=1.5, ls="--",
                  label="WT 95 % ellipse")
    ax.add_patch(ell)
    # Centroid markers
    ax.scatter([mu[0]], [mu[1]], marker="X", s=120, color="#4c78a8",
               edgecolor="black", linewidth=1, zorder=5)
    mu_mut = mut.mean(0)
    ax.scatter([mu_mut[0]], [mu_mut[1]], marker="X", s=120, color="#c0504d",
               edgecolor="black", linewidth=1, zorder=5)
    # Connecting line
    ax.plot([mu[0], mu_mut[0]], [mu[1], mu_mut[1]],
            color="#444", lw=1.8, ls="-")
    title = (f"{gene} {mutation}\n"
             f"Δ={result['delta_latent']:.1f}, Mahal={result['mahalanobis_sigma']:.2f}σ, "
             f"p={result['perm_pvalue']:.4f}")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("z0"); ax.set_ylabel("z1")
    ax.legend(fontsize=8, loc="best", frameon=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent-csv", required=True, type=Path,
                    help="v9 latent CSV with z0,z1,chain_key,gene columns")
    ap.add_argument("--mutations-csv", required=True, type=Path,
                    help="v9_chain_mutations.csv with annotation columns")
    ap.add_argument("--mutation-list-csv", required=True, type=Path,
                    help="CSV with at least gene,mutation columns")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    lat = pd.read_csv(args.latent_csv, keep_default_na=False)
    lat["chain_key"] = lat["chain_key"].astype(str).str.upper()
    muts = pd.read_csv(args.mutations_csv, keep_default_na=False)
    muts["chain_key"] = muts["chain_key"].astype(str).str.upper()
    df = muts.merge(lat[["chain_key", "z0", "z1"]],
                    on="chain_key", how="left")
    df["z0"] = df["z0"].astype(float); df["z1"] = df["z1"].astype(float)
    df = df[df["z0"].notna() & df["z1"].notna()].copy()

    candidates = pd.read_csv(args.mutation_list_csv)
    print(f"Testing {len(candidates)} mutations")

    rows = []
    valid = []   # list of (gene, mutation, wt_arr, mut_arr, result)
    for _, c in candidates.iterrows():
        gene = c["gene"]; mut = c["mutation"]
        sub = df[df["gene"] == gene]
        wt = sub[~sub["has_any_mutation_annotation"]][["z0", "z1"]].to_numpy(float)
        mut_chains = find_chains_for_mutation(sub, gene, mut)
        mu_arr = mut_chains[["z0", "z1"]].to_numpy(float)
        if len(wt) < 3 or len(mu_arr) < 1:
            rows.append({"gene": gene, "mutation": mut, "n_wt": len(wt),
                         "n_mut": len(mu_arr),
                         "comment": "too few chains"})
            continue
        cov = np.cov(wt.T) if len(wt) > 1 else np.eye(2)
        mu_wt = wt.mean(0); mu_mu = mu_arr.mean(0)
        delta = float(np.linalg.norm(mu_wt - mu_mu))
        mahal_sq = mahalanobis_sq(mu_mu, mu_wt, cov)
        sigma_wt = float(np.sqrt(np.trace(cov)))    # combined std
        pval, _ = permutation_pvalue(wt, mu_arr, n_perm=args.n_perm)
        boot_med, boot_lo, boot_hi = bootstrap_delta(
            wt, mu_arr, n_boot=args.n_boot)
        rows.append({
            "gene": gene, "mutation": mut,
            "n_wt": len(wt), "n_mut": len(mu_arr),
            "delta_latent": delta,
            "sigma_wt_combined": sigma_wt,
            "delta_in_wt_sigmas": delta / sigma_wt if sigma_wt > 0 else float("nan"),
            "mahalanobis_sq": mahal_sq,
            "mahalanobis_sigma": float(np.sqrt(mahal_sq)),
            "perm_pvalue": pval,
            "boot_delta_median": boot_med,
            "boot_delta_2.5pct": boot_lo,
            "boot_delta_97.5pct": boot_hi,
            "significant_perm_p<0.05": pval < 0.05,
            "significant_mahal_>3sigma": mahal_sq > chi2.ppf(0.99, df=2),
        })
        valid.append((gene, mut, wt, mu_arr, rows[-1]))

    res_df = pd.DataFrame(rows)
    res_df.to_csv(args.out / "significance_summary.csv", index=False)
    print("\n========== SIGNIFICANCE SUMMARY ==========")
    cols = ["gene", "mutation", "n_wt", "n_mut", "delta_latent",
            "delta_in_wt_sigmas", "mahalanobis_sigma",
            "perm_pvalue", "boot_delta_2.5pct", "boot_delta_97.5pct",
            "significant_perm_p<0.05", "significant_mahal_>3sigma"]
    show = [c for c in cols if c in res_df.columns]
    print(res_df[show].to_string(index=False))

    # Scatter plot grid: one panel per tested mutation
    if valid:
        n = len(valid)
        ncol = min(3, n)
        nrow = (n + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(5.5 * ncol, 4.5 * nrow),
                                  squeeze=False)
        for i, (gene, mut, wt, mu_arr, result) in enumerate(valid):
            ax = axes[i // ncol, i % ncol]
            plot_one_mutation(ax, wt, mu_arr, gene, mut, result)
        for j in range(len(valid), nrow * ncol):
            axes[j // ncol, j % ncol].axis("off")
        fig.suptitle(
            "Mutation Δlatent vs WT spread — v9 baseline AE latent",
            fontsize=14, y=1.005)
        fig.tight_layout()
        fig.savefig(args.out / "significance_scatter.png", dpi=200,
                    bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
