"""Extract activation loops for kinase families excluded from v9 by the
narrow [AS]PE motif regex (notably EGFR / ERBB family and ALK / LTK
family), apply the SAME spline-fit + Kabsch flank-alignment pipeline as
build_v9_ca_spline.py, and encode through the trained v9 FoldingNet
checkpoint.

OUT-OF-DISTRIBUTION CAVEAT.  These chains were excluded from v9 training
because their C-terminal activation-loop anchor diverges from APE:

  - EGFR / ERBB family use ALE
  - ALK / LTK family use PPE

Their loop lengths (27 residues) are identical to BRAF, so a length-27
encoder CAN process them, but the model was never trained on their
specific Cα geometries.  The latent coordinates produced here are
INFORMATIVE for cross-checking drug pharmacology (because the encoder
recovers the canonical activation-loop conformational classes when
generalised to many other kinases, e.g. the Clayton/Shen MD validation),
but they should be treated as PROVISIONAL until a v9.1 model is
retrained on the extended dataset.

Usage (on coulomb):
  /home/edina/miniforge3/envs/kinase_ae/bin/python \
      extract_and_encode_addendum.py \
      --kincore-fasta  /home/edina/kinase_v4_training/PK_labels_PDB.fasta \
      --pdb-dir        /home/edina/kinase_v4_training/PDBs \
      --ref-pdb        manuscript_draft/data/v9_lgbm_shap/figures/6UAN_full.pdb \
      --checkpoint     /home/edina/kinase_v4_training/v9_release/best.ckpt \
      --target-genes   EGFR,ERBB2,ERBB3,ERBB4,ALK,LTK \
      --out-csv        manuscript_draft/data/v9_lgbm_shap/per_drug_analysis/v9_addendum_latent.csv
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Re-use existing v9 helpers (we want byte-identical maths).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_v9_ca_spline import (
    BACKBONE, read_backbone, kabsch, spline_ca_arclen,
)


THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # protonation variants (rare in PDB but harmless)
    "HID": "H", "HIE": "H", "HIP": "H",
}

# Broadened activation-loop anchor regex (vs the original [AS]PE which
# excludes EGFR/ALK families).
DFG_RE = re.compile(r"D[FYL]G")
ANCHOR_RE = re.compile(r"[AYSTP][LP]E")


def parse_kincore_targets(fasta_path: Path,
                          target_genes: set[str]) -> dict[str, dict]:
    """Return chain_key.upper() -> {gene, dfg_spatial, dihedral, ligands}."""
    out = {}
    with fasta_path.open() as f:
        for line in f:
            if not line.startswith(">"):
                continue
            parts = line.rstrip().split("\t")
            ident = parts[0][1:]
            if ident.startswith("AF-"):
                continue
            if len(parts) < 2 or parts[1] not in target_genes:
                continue
            out[ident.upper()] = {
                "gene":        parts[1],
                "group":       parts[3] if len(parts) > 3 else "",
                "dfg_spatial": parts[4] if len(parts) > 4 else "",
                "dihedral":    parts[5] if len(parts) > 5 else "",
                "ligands_raw": parts[6] if len(parts) > 6 else "",
                "ligand_type": parts[7] if len(parts) > 7 else "",
            }
    return out


def pdb_chain_seq_and_resids(pdb_path: Path, chain_id: str
                             ) -> tuple[str, list[int]]:
    """Read CA atoms for the chain; return (one-letter seq, resid list)."""
    seq, resids = [], []
    seen_resids = set()
    with pdb_path.open() as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[21] != chain_id:
                continue
            if line[12:16].strip() != "CA":
                continue
            resn = line[17:20].strip()
            if resn not in THREE_TO_ONE:
                continue
            try:
                resi = int(line[22:26])
            except ValueError:
                continue
            if resi in seen_resids:
                continue   # skip altloc duplicates
            seen_resids.add(resi)
            seq.append(THREE_TO_ONE[resn])
            resids.append(resi)
    return "".join(seq), resids


def find_loop_in_pdb(seq: str, resids: list[int],
                     min_len: int = 14, max_len: int = 40,
                     target_len: int = 27
                     ) -> tuple[int, int, int, int] | None:
    """Find the DFG and anchor (APE/ALE/PPE/...) positions in the
    PDB-derived sequence.  Return (dfg_idx_seq, dfg_resi, anchor_idx_seq,
    anchor_E_resi) or None if no suitable pair found.

    anchor_E_resi is the PDB residue of the E (glutamate) of the
    APE/ALE/PPE anchor i.e. the last residue of the activation loop.
    """
    dfgs = [m.start() for m in DFG_RE.finditer(seq)]
    anchors = [m.start() for m in ANCHOR_RE.finditer(seq)]
    best = None
    best_score = None
    for d in dfgs:
        for a in anchors:
            gap = a - d
            if not (min_len <= gap <= max_len):
                continue
            score = abs(gap - target_len)
            if best_score is None or score < best_score:
                best, best_score = (d, a), score
    if best is None:
        return None
    d_idx, a_idx = best
    # The anchor pattern is 3 residues long (e.g. APE / ALE / PPE);
    # the Glu (E) is the 3rd character at offset +2.
    anchor_e_idx = a_idx + 2
    if anchor_e_idx >= len(resids):
        return None
    return d_idx, resids[d_idx], anchor_e_idx, resids[anchor_e_idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kincore-fasta", required=True, type=Path)
    ap.add_argument("--pdb-dir", required=True, type=Path)
    ap.add_argument("--ref-pdb", required=True, type=Path)
    ap.add_argument("--ref-chain", default="C")
    ap.add_argument("--ref-dfg", type=int, default=594)
    ap.add_argument("--ref-ape", type=int, default=623,
                    help="reference C-terminal anchor E residue (APE-E)")
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--target-genes", required=True,
                    help="comma-separated list of HUGO symbols, e.g. EGFR,ERBB2,ALK")
    ap.add_argument("--out-csv", required=True, type=Path)
    ap.add_argument("--flank", type=int, default=40)
    ap.add_argument("--min-flank-frac", type=float, default=0.7)
    ap.add_argument("--min-loop-frac", type=float, default=0.7)
    ap.add_argument("--flank-rmsd-max", type=float, default=8.0)
    ap.add_argument("--n-loop-points", type=int, default=27)
    args = ap.parse_args()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    target_set = {g.strip().upper() for g in args.target_genes.split(",")}
    print(f"Target genes: {sorted(target_set)}")

    # ---- Kincore target rows ----
    meta = parse_kincore_targets(args.kincore_fasta, target_set)
    print(f"Kincore chains with target gene: {len(meta)}")

    # ---- reference flank from 6UAN ----
    ref_bb = read_backbone(args.ref_pdb, args.ref_chain)
    if ref_bb is None:
        raise SystemExit(f"could not read reference flank from {args.ref_pdb}")
    ref_specs, ref_flank = [], []
    for r in range(args.ref_dfg - args.flank, args.ref_dfg):
        if r in ref_bb and "CA" in ref_bb[r]:
            ref_specs.append(("dfg", r - args.ref_dfg))
            ref_flank.append(ref_bb[r]["CA"])
    for r in range(args.ref_ape + 1, args.ref_ape + args.flank + 1):
        if r in ref_bb and "CA" in ref_bb[r]:
            ref_specs.append(("ape", r - args.ref_ape))
            ref_flank.append(ref_bb[r]["CA"])
    ref_flank = np.asarray(ref_flank, dtype=np.float32)
    print(f"reference flank atoms: {len(ref_specs)}/{2 * args.flank}")

    # ---- v9 encoder ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from molearn.models.small_foldingnet import Small_AutoEncoder
    net = Small_AutoEncoder(out_points=args.n_loop_points).to(device)
    state = torch.load(str(args.checkpoint), map_location=device,
                       weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        net.load_state_dict(state["model_state_dict"])
    elif isinstance(state, dict) and "state_dict" in state:
        net.load_state_dict(state["state_dict"])
    else:
        net.load_state_dict(state)
    net.eval()
    print(f"loaded checkpoint onto {device}")

    rows = []
    failures = []
    for chain_key, info in tqdm(meta.items()):
        pdb_id = chain_key[:4]
        chain_id = chain_key[4:]
        pdb_path = args.pdb_dir / f"{pdb_id}.pdb"
        if not pdb_path.exists():
            failures.append({"chain_key": chain_key,
                             "status": "no_pdb_file"})
            continue
        try:
            seq, resids = pdb_chain_seq_and_resids(pdb_path, chain_id)
        except Exception as e:
            failures.append({"chain_key": chain_key,
                             "status": f"pdb_read_exc:{e}"})
            continue
        if len(seq) < 100:
            failures.append({"chain_key": chain_key,
                             "status": f"chain_too_short:{len(seq)}"})
            continue
        loop = find_loop_in_pdb(seq, resids, target_len=args.n_loop_points)
        if loop is None:
            failures.append({"chain_key": chain_key,
                             "status": "no_motif_pair"})
            continue
        dfg_idx, dfg_resi, anchor_e_idx, anchor_e_resi = loop

        # ---- pull flanks + loop CAs from the chain backbone ----
        bb = read_backbone(pdb_path, chain_id)
        if bb is None:
            failures.append({"chain_key": chain_key,
                             "status": "backbone_read_failed"})
            continue
        # flank Cα: 40 N-term of DFG-D + 40 C-term of anchor-E
        flank_mob, flank_ref = [], []
        for (kind, off), ref_xyz in zip(ref_specs, ref_flank):
            if kind == "dfg":
                t = dfg_resi + off
            else:
                t = anchor_e_resi + off
            if t in bb and "CA" in bb[t]:
                flank_mob.append(bb[t]["CA"])
                flank_ref.append(ref_xyz)
        if len(flank_mob) < args.min_flank_frac * len(ref_specs):
            failures.append({"chain_key": chain_key,
                             "status": f"underresolved_flank:{len(flank_mob)}"})
            continue
        mob = np.asarray(flank_mob, dtype=np.float32)
        ref = np.asarray(flank_ref, dtype=np.float32)
        R, _, rmsd = kabsch(mob, ref)
        if rmsd > args.flank_rmsd_max:
            failures.append({"chain_key": chain_key,
                             "status": f"flank_rmsd_too_high:{rmsd:.2f}"})
            continue
        mc = mob.mean(axis=0); rc = ref.mean(axis=0)

        def transform(xyz):
            return (xyz - mc) @ R.T + rc

        loop_resis = list(range(dfg_resi, anchor_e_resi + 1))
        K = len(loop_resis)
        ca_present = []
        for r in loop_resis:
            if r in bb and "CA" in bb[r]:
                ca_present.append(transform(bb[r]["CA"]))
        if len(ca_present) < args.min_loop_frac * K:
            failures.append({"chain_key": chain_key,
                             "status": f"underresolved_loop:{len(ca_present)}/{K}"})
            continue
        ca_arr = np.asarray(ca_present, dtype=np.float32)
        spline = spline_ca_arclen(ca_arr, args.n_loop_points)
        if spline is None:
            failures.append({"chain_key": chain_key,
                             "status": "ca_spline_failed"})
            continue
        # centre to centroid -- matches training-time centering
        spline = spline - spline.mean(axis=0)

        # ---- encode + decode + Chamfer recon RMSD ----
        # FoldingNet decoder outputs 32 grid points; use Chamfer distance
        # (same as training loss) for the reconstruction-quality metric.
        with torch.no_grad():
            X = torch.from_numpy(spline[np.newaxis]).float().to(device)
            z = net.encode(X)
            recon = net.decode(z)
            if recon.shape[1] == 3:
                recon = recon.permute(0, 2, 1)
            X_c     = X     - X.mean(dim=1, keepdim=True)
            recon_c = recon - recon.mean(dim=1, keepdim=True)
            d2 = ((X_c.unsqueeze(2) - recon_c.unsqueeze(1)) ** 2).sum(-1)
            d_fwd = d2.min(dim=2).values.sqrt().mean(dim=1)
            d_bwd = d2.min(dim=1).values.sqrt().mean(dim=1)
            recon_rmsd = float((0.5 * (d_fwd + d_bwd)).item())
            z = z.cpu().numpy().reshape(-1)
        rows.append({
            "chain_key":     chain_key,
            "pdb":           pdb_id,
            "chain":         chain_id,
            "gene":          info["gene"],
            "group":         info["group"],
            "dfg_spatial":   info["dfg_spatial"],
            "dihedral":      info["dihedral"],
            "ligand_type":   info["ligand_type"],
            "ligands_raw":   info["ligands_raw"],
            "anchor_motif":  seq[anchor_e_idx - 2:anchor_e_idx + 1],
            "loop_length":   K,
            "flank_rmsd":    float(rmsd),
            "recon_rmsd":    recon_rmsd,
            "n_loop_present": len(ca_present),
            "ood":           True,
            "z0":            float(z[0]),
            "z1":            float(z[1]),
        })

    df_out = pd.DataFrame(rows)
    df_fail = pd.DataFrame(failures)
    df_out.to_csv(args.out_csv, index=False)
    print(f"\nWrote {args.out_csv}  ({len(df_out)} chains encoded)")
    fail_csv = args.out_csv.parent / (args.out_csv.stem + "_failures.csv")
    df_fail.to_csv(fail_csv, index=False)
    print(f"Failures: {len(df_fail)} (logged to {fail_csv})")
    if len(df_out):
        print()
        print("Per-gene yield:")
        print(df_out.groupby("gene")["chain_key"].count().to_string())
        print()
        print("Anchor motifs found:")
        print(df_out["anchor_motif"].value_counts().to_string())
        print()
        print("Median flank RMSD:", df_out["flank_rmsd"].median())


if __name__ == "__main__":
    main()
