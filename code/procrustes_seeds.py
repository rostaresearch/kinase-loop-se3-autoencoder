"""R2: are the latent AXES identifiable across random seeds?

A 2-D autoencoder latent is determined only up to reparameterisation: equal
reconstruction error does not imply the individual axes, their signs or their
feature rankings are preserved. This script quantifies that.

Part A -- geometry.  Is the *configuration* reproducible (rotation/reflection
aside)?  Measured two ways: the rotation-invariant correlation between seeds'
pairwise-distance matrices, and the orthogonal-Procrustes disparity after
aligning each seed to the reference.

Part B -- axes.  Are the raw axes comparable across seeds before alignment,
and does Procrustes alignment fix it?

Part C -- feature rankings.  The paper makes axis-specific claims ("499-641
drives z0", "525-577 drives z1").  Refit LightGBM importance per seed on the
RAW axes and on the PROCRUSTES-ALIGNED axes, and see which top features survive.
"""
import numpy as np, pandas as pd, itertools
from pathlib import Path
from scipy.linalg import orthogonal_procrustes
from scipy.stats import spearmanr
from lightgbm import LGBMRegressor

S = "/home/edina/kinase_v91_share"
D = Path("/home/edina/leakage")
SD = Path("/home/edina/seedstab")
SEEDS = [25, 101, 202, 303, 404]
REF = 25

manifest = pd.read_csv(f"{S}/data/manifest_v91.csv", keep_default_na=False)
manifest["chain_key"] = manifest["chain_key"].astype(str).str.upper()

Z = {}
for s in SEEDS:
    d = pd.read_csv(SD / f"latent_seed{s}.csv").sort_values("idx")
    assert len(d) == len(manifest), (s, len(d), len(manifest))
    Z[s] = d[["z0", "z1"]].to_numpy(float)

# sanity: does our CPU-retrained seed 25 reproduce the published latent?
pub = pd.read_csv(f"{S}/v91_SE3_latent_seed25.csv", keep_default_na=False)
pub["chain_key"] = pub["chain_key"].astype(str).str.upper()
pub = pub.set_index("chain_key").loc[manifest["chain_key"]][["z0", "z1"]].to_numpy(float)


def norm(A):
    """centre and scale to unit Frobenius norm (Procrustes convention)."""
    A = A - A.mean(0)
    return A / np.linalg.norm(A)


def align(A, B):
    """Orthogonal Procrustes: rotate/reflect A onto B. Returns aligned A, R, disparity."""
    An, Bn = norm(A), norm(B)
    R, _ = orthogonal_procrustes(An, Bn)
    return An @ R, R, float(np.linalg.norm(An @ R - Bn) ** 2)


def dm_corr(A, B, n=1500, rng=np.random.default_rng(0)):
    """rotation/reflection-invariant agreement: correlate pairwise distances."""
    i = rng.choice(len(A), n, replace=False)
    a, b = A[i], B[i]
    da = np.linalg.norm(a[:, None] - a[None], axis=-1)[np.triu_indices(n, 1)]
    db = np.linalg.norm(b[:, None] - b[None], axis=-1)[np.triu_indices(n, 1)]
    return float(spearmanr(da, db).statistic)


print("=" * 74)
print("PART A -- is the latent GEOMETRY reproducible across seeds?")
print("=" * 74)
Za, _, dpub = align(Z[REF], pub)
print(f"retrained seed{REF} vs published seed25: disparity {dpub:.4f}  "
      f"distance-matrix rho {dm_corr(Z[REF], pub):.3f}   (pipeline sanity check)")
rows = []
for a, b in itertools.combinations(SEEDS, 2):
    _, R, disp = align(Z[a], Z[b])
    rows.append(dict(seed_a=a, seed_b=b, procrustes_disparity=disp,
                     reflection=bool(np.linalg.det(R) < 0),
                     dist_matrix_rho=dm_corr(Z[a], Z[b])))
geo = pd.DataFrame(rows)
print(geo.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print(f"\nmean disparity {geo.procrustes_disparity.mean():.4f}  "
      f"mean distance-matrix rho {geo.dist_matrix_rho.mean():.3f}  "
      f"reflections needed in {geo.reflection.sum()}/{len(geo)} pairs")

print()
print("=" * 74)
print("PART B -- are the raw AXES comparable across seeds?")
print("=" * 74)
rows = []
for a, b in itertools.combinations(SEEDS, 2):
    Aa, _, _ = align(Z[a], Z[b])
    Bn = norm(Z[b])
    An = norm(Z[a])
    rows.append(dict(
        seed_a=a, seed_b=b,
        raw_z0=abs(np.corrcoef(An[:, 0], Bn[:, 0])[0, 1]),
        raw_z1=abs(np.corrcoef(An[:, 1], Bn[:, 1])[0, 1]),
        aligned_z0=abs(np.corrcoef(Aa[:, 0], Bn[:, 0])[0, 1]),
        aligned_z1=abs(np.corrcoef(Aa[:, 1], Bn[:, 1])[0, 1])))
ax = pd.DataFrame(rows)
print(ax.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print(f"\nmean |r| raw:     z0 {ax.raw_z0.mean():.3f}   z1 {ax.raw_z1.mean():.3f}")
print(f"mean |r| aligned: z0 {ax.aligned_z0.mean():.3f}   z1 {ax.aligned_z1.mean():.3f}")

# ---------------- Part C: do the axis-specific FEATURE claims survive? ------
print()
print("=" * 74)
print("PART C -- do the top predictive features survive across seeds?")
print("=" * 74)
X_raw = np.load(D / "X_raw.npy")
pairs = pd.read_csv(D / "candidate_pairs.csv")
keep = (~np.isnan(X_raw)).mean(axis=0) >= 0.75
X = X_raw[:, keep]
labels = [f"{r.resi_i}-{r.resi_j}" for r in pairs[keep].itertuples()]
frac = np.isnan(X).mean(axis=1)
col = np.nanmean(X, axis=0)
idx = np.where(np.isnan(X)); X[idx] = np.take(col, idx[1])
ok = frac <= 0.50
X = X[ok]
print(f"{X.shape[0]} chains x {X.shape[1]} features", flush=True)

def top_feats(y, k=5):
    m = LGBMRegressor(n_estimators=400, num_leaves=31, verbose=-1,
                      random_state=25, n_jobs=8).fit(X, y)
    o = np.argsort(m.booster_.feature_importance("gain"))[::-1][:k]
    return [labels[i] for i in o]

res = []
for s in SEEDS:
    raw = norm(Z[s])[ok]
    ali = (align(Z[s], Z[REF])[0])[ok]
    for axis in (0, 1):
        res.append(dict(seed=s, axis=f"z{axis}",
                        raw_top5=" ".join(top_feats(raw[:, axis])),
                        aligned_top5=" ".join(top_feats(ali[:, axis]))))
    print(f"  seed {s} done", flush=True)
fc = pd.DataFrame(res)
print(fc.to_string(index=False))

for axis in ("z0", "z1"):
    sub = fc[fc.axis == axis]
    r1_raw = set(x.split()[0] for x in sub.raw_top5)
    r1_ali = set(x.split()[0] for x in sub.aligned_top5)
    def jac(col):
        sets = [set(x.split()) for x in col]
        return np.mean([len(a & b) / len(a | b) for a, b in itertools.combinations(sets, 2)])
    print(f"\n{axis}: rank-1 feature across seeds  RAW={sorted(r1_raw)}   "
          f"ALIGNED={sorted(r1_ali)}")
    print(f"{axis}: mean pairwise top-5 Jaccard   raw {jac(sub.raw_top5):.2f}  "
          f"aligned {jac(sub.aligned_top5):.2f}")

geo.to_csv(D / "seed_geometry.csv", index=False)
ax.to_csv(D / "seed_axes.csv", index=False)
fc.to_csv(D / "seed_feature_stability.csv", index=False)
print("\nwrote seed_geometry.csv / seed_axes.csv / seed_feature_stability.csv")
