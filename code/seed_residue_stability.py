"""R2 follow-up: pair-level rankings are unstable across seeds -- is the
RESIDUE-level importance (what the manuscript figure actually shows) stable?

Aggregates each seed's per-pair gain to per-residue totals (a residue's
importance = sum of gain over every pair it participates in), then measures
cross-seed agreement on the Procrustes-aligned axes.
"""
import numpy as np, pandas as pd, itertools
from pathlib import Path
from scipy.linalg import orthogonal_procrustes
from scipy.stats import spearmanr
from lightgbm import LGBMRegressor

D = Path("/home/edina/leakage"); SD = Path("/home/edina/seedstab")
SEEDS = [25, 101, 202, 303, 404]; REF = 25

Z = {s: pd.read_csv(SD / f"latent_seed{s}.csv").sort_values("idx")[["z0", "z1"]].to_numpy(float)
     for s in SEEDS}

def norm(A):
    A = A - A.mean(0); return A / np.linalg.norm(A)

def align(A, B):
    An, Bn = norm(A), norm(B)
    R, _ = orthogonal_procrustes(An, Bn); return An @ R

X_raw = np.load(D / "X_raw.npy")
pairs = pd.read_csv(D / "candidate_pairs.csv")
keep = (~np.isnan(X_raw)).mean(axis=0) >= 0.75
X = X_raw[:, keep]
pk = pairs[keep].reset_index(drop=True)
frac = np.isnan(X).mean(axis=1)
col = np.nanmean(X, axis=0); idx = np.where(np.isnan(X)); X[idx] = np.take(col, idx[1])
ok = frac <= 0.50; X = X[ok]
resis = sorted(set(pk.resi_i) | set(pk.resi_j))
rpos = {r: i for i, r in enumerate(resis)}
Ai = pk.resi_i.map(rpos).to_numpy(); Aj = pk.resi_j.map(rpos).to_numpy()
print(f"{X.shape[0]} chains x {X.shape[1]} pairs over {len(resis)} residues", flush=True)

def residue_gain(y):
    m = LGBMRegressor(n_estimators=400, num_leaves=31, verbose=-1,
                      random_state=25, n_jobs=8).fit(X, y)
    g = m.booster_.feature_importance("gain").astype(float)
    out = np.zeros(len(resis))
    np.add.at(out, Ai, g); np.add.at(out, Aj, g)
    return out

RG = {}
zsd = {}
for s in SEEDS:
    A = align(Z[s], Z[REF])[ok]
    zsd[s] = (A[:, 0].std(), A[:, 1].std())
    for a in (0, 1):
        RG[(s, a)] = residue_gain(A[:, a])
    print(f"  seed {s} done  (aligned z0 sd {zsd[s][0]:.4f}, z1 sd {zsd[s][1]:.4f})", flush=True)

print("\n" + "=" * 70)
print("RESIDUE-LEVEL cross-seed agreement (Procrustes-aligned axes)")
print("=" * 70)
rows = []
for a, name in ((0, "z0"), (1, "z1")):
    rho = [spearmanr(RG[(x, a)], RG[(y, a)]).statistic
           for x, y in itertools.combinations(SEEDS, 2)]
    def jac(k):
        tops = [set(np.argsort(RG[(s, a)])[::-1][:k]) for s in SEEDS]
        return np.mean([len(u & v) / len(u | v) for u, v in itertools.combinations(tops, 2)])
    rows.append(dict(axis=name, spearman_mean=np.mean(rho), spearman_min=np.min(rho),
                     jaccard_top10=jac(10), jaccard_top20=jac(20), jaccard_top30=jac(30)))
    # which residues are top-10 in ALL seeds?
    tops = [set(np.array(resis)[np.argsort(RG[(s, a)])[::-1][:10]]) for s in SEEDS]
    core = sorted(set.intersection(*tops))
    union = sorted(set.union(*tops))
    print(f"\n{name}: residues in the top-10 of ALL 5 seeds: {core}")
    print(f"{name}: union of top-10 across seeds ({len(union)}): {union}")
tab = pd.DataFrame(rows)
print("\n" + tab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
tab.to_csv(D / "seed_residue_stability.csv", index=False)
pd.DataFrame({f"seed{s}_{'z0' if a==0 else 'z1'}": RG[(s, a)]
              for s in SEEDS for a in (0, 1)}, index=resis).to_csv(D / "seed_residue_gain.csv")
print("\nwrote seed_residue_stability.csv / seed_residue_gain.csv")
