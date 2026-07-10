"""Lit-search-based rescue extraction for the 1,917 kinase chains still
missing after the [APS][LP]E broadening.

Adds:
  - DWG to the DFG regex  ->  CK2 family (CSNK2A1, CSNK2A2)
  - AIE / AVE anchors     ->  MERTK / RET / AXL / TYRO3 receptor TKs
  - AWE anchor            ->  DDR1 / DDR2 collagen receptor TKs
  - TIE / TLE anchors     ->  VRK1 / VRK2 / TTBK1 / TTBK2 CK1-group kinases
  - PTE anchor            ->  HASPIN
  - MYE / YPE anchors     ->  TBK1 / IKKε family

Picks the anchor with gap closest to 27 in [14, 40], same logic as the
v9 pipeline.  Writes the same per-chain CSV format as
extract_and_encode_addendum.py with the SAME columns so the two outputs
can be merged.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_v9_ca_spline import BACKBONE, read_backbone, kabsch, spline_ca_arclen

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "HID": "H", "HIE": "H", "HIP": "H",
}

# Lit-search rescued regexes
#   DFG variants: include DWG (CK2) on top of standard D[FYL]G
DFG_RE = re.compile(r"D[FYLW]G")
#   Anchor variants observed in real kinase activation-loop ends:
#     [APS][LP]E  -- APE, SPE, ALE, PPE, PLE, SLE                (already covered)
#     AIE, AVE    -- MERTK/RET/AXL/TYRO3
#     AWE         -- DDR1/DDR2
#     TIE, TLE    -- VRK1/2, TTBK1/2
#     PTE         -- HASPIN
#     VPE, VLE    -- PKG, CDPK2
#
# IMPORTANT: M and Y are deliberately EXCLUDED from the first character.
# An earlier version of this regex included them, which caused MIE
# (Met-Ile-Glu) false-positive matches in AURKA -- the regex picked up
# the start of the alpha-EF helix immediately after the activation loop's
# real PPE anchor.  The MIE 'match' was structurally wrong (it landed
# the encoder one helix-turn past the real loop end) and gave high recon
# RMSDs.  Restricting first char to [APSTV] removes that family of
# helix-start FPs while preserving every real anchor variant we have
# evidence for.
ANCHOR_RE = re.compile(r"[APSTV][LIPWV]E")


def parse_kincore_targets(fasta_path: Path,
                          target_genes: set[str]) -> dict[str, dict]:
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
    seq, resids = [], []
    seen = set()
    with pdb_path.open() as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[21] != chain_id:
                continue
            if line[12:16].strip() != "CA":
                continue
            rn = line[17:20].strip()
            if rn not in THREE_TO_ONE:
                continue
            try:
                resi = int(line[22:26])
            except ValueError:
                continue
            if resi in seen:
                continue
            seen.add(resi)
            seq.append(THREE_TO_ONE[rn])
            resids.append(resi)
    return "".join(seq), resids


def find_loop(seq: str, resids: list[int]):
    """Return (dfg_idx, dfg_resi, anchor_e_idx, anchor_e_resi, anchor_motif)
    or None.  Tries broader DFG + broader anchor regexes, picks the gap
    closest to 27.
    """
    dfgs = [m.start() for m in DFG_RE.finditer(seq)]
    anchors = [m.start() for m in ANCHOR_RE.finditer(seq)]
    best, best_score = None, None
    for d in dfgs:
        for a in anchors:
            gap = a - d
            if not (14 <= gap <= 40):
                continue
            score = abs(gap - 27)
            if best_score is None or score < best_score:
                best, best_score = (d, a), score
    if best is None:
        return None
    d_idx, a_idx = best
    e_idx = a_idx + 2
    if e_idx >= len(resids):
        return None
    return (d_idx, resids[d_idx], e_idx, resids[e_idx],
            seq[a_idx:a_idx + 3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kincore-fasta", required=True, type=Path)
    ap.add_argument("--pdb-dir", required=True, type=Path)
    ap.add_argument("--ref-pdb", required=True, type=Path)
    ap.add_argument("--ref-chain", default="C")
    ap.add_argument("--ref-dfg", type=int, default=594)
    ap.add_argument("--ref-ape", type=int, default=623)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--target-genes", required=True,
                    help="comma-separated list of HUGO symbols")
    ap.add_argument("--out-csv", required=True, type=Path)
    ap.add_argument("--flank", type=int, default=40)
    ap.add_argument("--min-flank-frac", type=float, default=0.7)
    ap.add_argument("--min-loop-frac", type=float, default=0.7)
    ap.add_argument("--flank-rmsd-max", type=float, default=8.0)
    ap.add_argument("--n-loop-points", type=int, default=27)
    args = ap.parse_args()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    target_set = {g.strip().upper() for g in args.target_genes.split(",")}
    print(f"Target genes ({len(target_set)})")
    meta = parse_kincore_targets(args.kincore_fasta, target_set)
    print(f"Kincore chains with target gene: {len(meta)}")

    ref_bb = read_backbone(args.ref_pdb, args.ref_chain)
    if ref_bb is None:
        raise SystemExit(f"could not read reference {args.ref_pdb}")
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
    print(f"ref flank atoms: {len(ref_specs)}/{2 * args.flank}")

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
            failures.append({"chain_key": chain_key, "status": "no_pdb_file"})
            continue
        try:
            seq, resids = pdb_chain_seq_and_resids(pdb_path, chain_id)
        except Exception as e:
            failures.append({"chain_key": chain_key, "status": f"pdb_read_exc:{e}"})
            continue
        if len(seq) < 100:
            failures.append({"chain_key": chain_key,
                             "status": f"chain_too_short:{len(seq)}"})
            continue
        loop = find_loop(seq, resids)
        if loop is None:
            failures.append({"chain_key": chain_key, "status": "no_motif_pair"})
            continue
        dfg_idx, dfg_resi, anchor_e_idx, anchor_e_resi, anchor_motif = loop

        bb = read_backbone(pdb_path, chain_id)
        if bb is None:
            failures.append({"chain_key": chain_key,
                             "status": "backbone_read_failed"})
            continue
        flank_mob, flank_ref = [], []
        for (kind, off), ref_xyz in zip(ref_specs, ref_flank):
            t = (dfg_resi + off) if kind == "dfg" else (anchor_e_resi + off)
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
            failures.append({"chain_key": chain_key, "status": "ca_spline_failed"})
            continue
        spline = spline - spline.mean(axis=0)

        with torch.no_grad():
            X = torch.from_numpy(spline[np.newaxis]).float().to(device)
            z = net.encode(X)
            recon = net.decode(z)
            if recon.shape[1] == 3:
                recon = recon.permute(0, 2, 1)
            X_c = X - X.mean(dim=1, keepdim=True)
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
            "anchor_motif":  anchor_motif,
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
    fail_csv = args.out_csv.parent / (args.out_csv.stem + "_failures.csv")
    df_fail.to_csv(fail_csv, index=False)
    print(f"\nWrote {args.out_csv}  ({len(df_out)} chains rescued)")
    print(f"Failures: {len(df_fail)}")
    if len(df_out):
        print()
        print("Per-gene rescue yield:")
        print(df_out.groupby("gene")["chain_key"].count().to_string())
        print()
        print("Anchor motifs found:")
        print(df_out["anchor_motif"].value_counts().to_string())
        print()
        print(f"Median flank RMSD: {df_out['flank_rmsd'].median():.2f} Å")
        print(f"Median recon RMSD: {df_out['recon_rmsd'].median():.2f} Å")


if __name__ == "__main__":
    main()
