"""R1: fit ALL preprocessing inside the training fold.

The published pipeline selects which candidate pairs pass the >=75% coverage
filter, and computes the column means used for imputation, over the WHOLE
dataset before splitting. Both steps see the test chains, so the reported R2 is
optimistic. Here the pair filter and the imputation means are refit on the
training rows of each fold only.

Runs each of the three validation designs (random / grouped by PDB entry /
grouped by gene) twice on IDENTICAL folds -- once leaky, once nested -- so the
difference is attributable to the preprocessing and nothing else.
"""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor

S = "/home/edina/kinase_v91_share"
D = Path("/home/edina/leakage")
MIN_PAIR_COV = 0.75      # published pair-coverage threshold
MAX_CHAIN_IMP = 0.50     # published per-chain "mostly imputed" drop threshold

X_raw = np.load(D / "X_raw.npy")                       # (6531, 8128), NaN = missing
manifest = pd.read_csv(D / "manifest_kept.csv", keep_default_na=False)
lat = pd.read_csv(f"{S}/v91_SE3_latent_seed25.csv", keep_default_na=False)
lat["chain_key"] = lat["chain_key"].astype(str).str.upper()
lat = lat.set_index("chain_key")

keys = manifest["chain_key"].astype(str).str.upper().values
has_lat = np.array([k in lat.index for k in keys])
X_raw, keys, manifest = X_raw[has_lat], keys[has_lat], manifest[has_lat].reset_index(drop=True)
Y = lat.loc[list(keys), ["z0", "z1"]].to_numpy(float)
print(f"chains with latent = {len(X_raw)}  candidate pairs = {X_raw.shape[1]}", flush=True)


def fit_prep(Xtr_raw):
    """Return (kept pair mask, column means) fit on training rows ONLY."""
    keep = (~np.isnan(Xtr_raw)).mean(axis=0) >= MIN_PAIR_COV
    means = np.nanmean(Xtr_raw[:, keep], axis=0)
    means = np.where(np.isnan(means), 0.0, means)   # pair unseen in train
    return keep, means


def apply_prep(X_raw_sub, keep, means):
    Xs = X_raw_sub[:, keep].copy()
    frac = np.isnan(Xs).mean(axis=1)
    idx = np.where(np.isnan(Xs))
    Xs[idx] = np.take(means, idx[1])
    return Xs, frac


def fit_predict(Xtr, Ytr, Xte):
    return np.column_stack([
        LGBMRegressor(n_estimators=400, num_leaves=31, verbose=-1,
                      random_state=25, n_jobs=8).fit(Xtr, Ytr[:, j]).predict(Xte)
        for j in range(2)])


def combined_r2(Yte, pred):
    ss = ((Yte - pred) ** 2).sum()
    tot = ((Yte - Yte.mean(0)) ** 2).sum()
    return 1 - ss / tot


# ---- LEAKY preprocessing: fit once on everything, exactly as published ----
keep_all, means_all = fit_prep(X_raw)
X_leaky, frac_all = apply_prep(X_raw, keep_all, means_all)
ok_all = frac_all <= MAX_CHAIN_IMP
print(f"leaky prep: {keep_all.sum()} pairs, {ok_all.sum()} chains pass the "
      f"<={MAX_CHAIN_IMP:.0%} imputed filter", flush=True)

designs = {
    "random":    None,
    "PDB entry": np.array([k[:4] for k in keys]),
    "gene":      manifest["gene"].astype(str).values,
}

rows = []
for name, g in designs.items():
    cv = KFold(5, shuffle=True, random_state=25) if g is None else GroupKFold(5)
    splits = list(cv.split(X_raw) if g is None else cv.split(X_raw, groups=g))
    leaky_r2, nested_r2, npairs = [], [], []

    for fold, (tr, te) in enumerate(splits):
        # -- leaky: global prep, then restrict to the published chain filter
        tr_l = tr[ok_all[tr]]; te_l = te[ok_all[te]]
        pred = fit_predict(X_leaky[tr_l], Y[tr_l], X_leaky[te_l])
        leaky_r2.append(combined_r2(Y[te_l], pred))

        # -- nested: prep fit on this fold's training rows only
        keep, means = fit_prep(X_raw[tr])
        Xtr, ftr = apply_prep(X_raw[tr], keep, means)
        Xte, fte = apply_prep(X_raw[te], keep, means)
        tr_ok = ftr <= MAX_CHAIN_IMP; te_ok = fte <= MAX_CHAIN_IMP
        pred = fit_predict(Xtr[tr_ok], Y[tr][tr_ok], Xte[te_ok])
        nested_r2.append(combined_r2(Y[te][te_ok], pred))
        npairs.append(int(keep.sum()))
        print(f"  {name:10s} fold {fold}: leaky {leaky_r2[-1]:.3f}  "
              f"nested {nested_r2[-1]:.3f}  ({keep.sum()} pairs, "
              f"{te_ok.sum()}/{len(te)} test chains kept)", flush=True)

    l, n_ = np.array(leaky_r2), np.array(nested_r2)
    rows.append(dict(design=name,
                     leaky_mean=l.mean(), leaky_sd=l.std(),
                     nested_mean=n_.mean(), nested_sd=n_.std(),
                     delta=n_.mean() - l.mean(),
                     pairs_min=min(npairs), pairs_max=max(npairs),
                     leaky_folds=" ".join(f"{v:.3f}" for v in l),
                     nested_folds=" ".join(f"{v:.3f}" for v in n_)))
    print(f"== {name:10s} leaky {l.mean():.3f}+/-{l.std():.3f}   "
          f"nested {n_.mean():.3f}+/-{n_.std():.3f}   "
          f"delta {n_.mean()-l.mean():+.3f}", flush=True)

out = pd.DataFrame(rows)
out.to_csv(D / "nested_cv_results.csv", index=False)
print("\n=== R1 FINAL ===")
print(out[["design", "leaky_mean", "leaky_sd", "nested_mean", "nested_sd", "delta"]]
      .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
