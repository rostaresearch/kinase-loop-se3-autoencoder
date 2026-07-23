"""C4: is the random chain split pseudoreplicated? Compare random vs grouped CV."""
import sys, numpy as np, pandas as pd
sys.path.insert(0,"/home/edina/kinase_v91_share/figure_pipeline")
from predict_v9_lgbm_shap import build_distance_matrix, norm_key
from sklearn.model_selection import GroupKFold, KFold
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
from pathlib import Path

S="/home/edina/kinase_v91_share"
X, manifest, pairs, frac = build_distance_matrix(
    Path(f"{S}/data/v9_1_braf_mapped_conserved_residues.csv"),
    Path(f"{S}/data/manifest_v91.csv"), Path(f"{S}/data/PDBs"),
    "6UANC", 9999, 0.75)
lat=pd.read_csv(f"{S}/v91_SE3_latent_seed25.csv",keep_default_na=False)
lat["chain_key"]=lat["chain_key"].astype(str).str.upper(); lat=lat.set_index("chain_key")
keys=manifest["chain_key"].astype(str).str.upper().values
keep=np.array([k in lat.index for k in keys]) & (frac<=0.5)
X=X[keep]; keys=keys[keep]; man=manifest[keep].reset_index(drop=True)
Y=lat.loc[list(keys),["z0","z1"]].to_numpy(float)
col=np.nanmean(X,axis=0); idx=np.where(np.isnan(X)); X[idx]=np.take(col,idx[1])
print(f"chains={len(X)} features={X.shape[1]}", flush=True)

groups={"random":None,
        "PDB entry":np.array([k[:4] for k in keys]),
        "gene":man["gene"].astype(str).values}
for name,g in groups.items():
    cv = KFold(5,shuffle=True,random_state=25) if g is None else GroupKFold(5)
    r2s=[]
    for tr,te in (cv.split(X) if g is None else cv.split(X,groups=g)):
        pr=np.column_stack([LGBMRegressor(n_estimators=400,num_leaves=31,verbose=-1,random_state=25)
                            .fit(X[tr],Y[tr,j]).predict(X[te]) for j in range(2)])
        ss=((Y[te]-pr)**2).sum(); tot=((Y[te]-Y[te].mean(0))**2).sum()
        r2s.append(1-ss/tot)
    r2s=np.array(r2s)
    print(f"  {name:10s} R2 = {r2s.mean():.3f} +/- {r2s.std():.3f}   folds={np.round(r2s,3)}", flush=True)
