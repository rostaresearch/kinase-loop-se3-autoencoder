"""Build the v9.1 training dataset:  v9 in-distribution chains (5,318)
+ all OOD chains we successfully re-encoded (1,213) = 6,531 chains in
a single combined CA-only PDB ready for FoldingNet retraining.

For the v9 in-distribution rows we already have spline-fit Cα coords
in ``combined_v9_ca.pdb``.  For the OOD rows we re-run the
addendum-extractor's spline-fit logic (broadened regex
``[APSTV][LIPWV]E`` for the anchor) and append the resulting 27-Cα
coords as new MODELs.

Outputs (in --out-dir):
  combined_v91_ca.pdb        6,531 MODELs (27 CAs each)
  manifest_v91.csv           chain_key, gene, group, dfg_spatial,
                              dihedral, ligand_type, flank_rmsd,
                              loop_length, anchor_motif, source
  train_idx.txt / test_idx.txt   90/10 split.  v9-indist gets the
                                 same indices it had in v9_release;
                                 new OOD rows are appended and
                                 randomly assigned 90/10.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_v9_ca_spline import (
    BACKBONE, read_backbone, kabsch, spline_ca_arclen
)


THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "HID": "H", "HIE": "H", "HIP": "H",
}

# Broadened regex matching what produced our merged OOD set.
DFG_RE = re.compile(r"D[FYLW]G")
# Excluding M (which gave MIE false positives) and Y from 1st char.
ANCHOR_RE = re.compile(r"[APSTV][LIPWV]E")


def pdb_chain_seq_resids(pdb_path: Path, chain_id: str):
    seq, resids, seen = [], [], set()
    with pdb_path.open() as f:
        for line in f:
            if not line.startswith("ATOM"): continue
            if line[21] != chain_id: continue
            if line[12:16].strip() != "CA": continue
            rn = line[17:20].strip()
            if rn not in THREE_TO_ONE: continue
            try: resi = int(line[22:26])
            except ValueError: continue
            if resi in seen: continue
            seen.add(resi)
            seq.append(THREE_TO_ONE[rn]); resids.append(resi)
    return "".join(seq), resids


def find_loop(seq, resids):
    dfgs = [m.start() for m in DFG_RE.finditer(seq)]
    anchors = [m.start() for m in ANCHOR_RE.finditer(seq)]
    best, best_score = None, None
    for d in dfgs:
        for a in anchors:
            gap = a - d
            if not (14 <= gap <= 40): continue
            score = abs(gap - 27)
            if best_score is None or score < best_score:
                best, best_score = (d, a, gap), score
    if best is None: return None
    d_idx, a_idx, _ = best
    e_idx = a_idx + 2
    if e_idx >= len(resids): return None
    return d_idx, resids[d_idx], e_idx, resids[e_idx], seq[a_idx:a_idx + 3]


def load_v9_indist_models(pdb_path: Path, manifest_csv: Path
                          ) -> dict[str, np.ndarray]:
    """Map chain_key -> (27, 3) Cα coords from the existing combined_v9_ca.pdb."""
    manifest = pd.read_csv(manifest_csv, keep_default_na=False)
    print(f"manifest_v9: {len(manifest)} rows")
    chain_keys = manifest["chain_key"].astype(str).str.upper().tolist()
    models = []
    cur = []
    with pdb_path.open() as f:
        for line in f:
            if line.startswith("MODEL"):
                cur = []
            elif line.startswith("ATOM") and line[12:16].strip() == "CA":
                cur.append([float(line[30:38]),
                            float(line[38:46]),
                            float(line[46:54])])
            elif line.startswith("ENDMDL") and cur:
                models.append(np.asarray(cur, dtype=np.float32))
                cur = []
    assert len(models) == len(chain_keys), (
        f"PDB has {len(models)} MODELs but manifest has {len(chain_keys)} rows")
    return {k: m for k, m in zip(chain_keys, models)}


def extract_new_chain(chain_key: str, pdb_dir: Path, ref_specs,
                      ref_flank, args):
    pdb_id, chain_id = chain_key[:4], chain_key[4:]
    pdb_path = pdb_dir / f"{pdb_id}.pdb"
    if not pdb_path.exists():
        return None, "no_pdb"
    seq, resids = pdb_chain_seq_resids(pdb_path, chain_id)
    if len(seq) < 100:
        return None, f"chain_short_{len(seq)}"
    loop = find_loop(seq, resids)
    if loop is None:
        return None, "no_motif"
    d_idx, dfg_resi, e_idx, ape_resi, anchor_motif = loop

    bb = read_backbone(pdb_path, chain_id)
    if bb is None:
        return None, "no_backbone"
    flank_mob, flank_ref = [], []
    for (kind, off), ref_xyz in zip(ref_specs, ref_flank):
        t = (dfg_resi + off) if kind == "dfg" else (ape_resi + off)
        if t in bb and "CA" in bb[t]:
            flank_mob.append(bb[t]["CA"])
            flank_ref.append(ref_xyz)
    if len(flank_mob) < args.min_flank_frac * len(ref_specs):
        return None, f"flank_low_{len(flank_mob)}"
    mob = np.asarray(flank_mob, dtype=np.float32)
    ref = np.asarray(flank_ref, dtype=np.float32)
    R, _, rmsd = kabsch(mob, ref)
    if rmsd > args.flank_rmsd_max:
        return None, f"flank_rmsd_{rmsd:.2f}"
    mc = mob.mean(0); rc = ref.mean(0)

    def transform(xyz): return (xyz - mc) @ R.T + rc

    loop_resis = list(range(dfg_resi, ape_resi + 1))
    K = len(loop_resis)
    ca_present = []
    for r in loop_resis:
        if r in bb and "CA" in bb[r]:
            ca_present.append(transform(bb[r]["CA"]))
    if len(ca_present) < args.min_loop_frac * K:
        return None, f"loop_low_{len(ca_present)}/{K}"
    ca_arr = np.asarray(ca_present, dtype=np.float32)
    spline = spline_ca_arclen(ca_arr, 27)
    if spline is None:
        return None, "spline_fail"
    spline = spline - spline.mean(0)
    return {
        "coords": spline.astype(np.float32),
        "anchor_motif": anchor_motif,
        "loop_length": K,
        "flank_rmsd": float(rmsd),
    }, "ok"


def write_ca_pdb(handle, coords, model_idx):
    handle.write(f"MODEL {model_idx}\n")
    for k, p in enumerate(coords, start=1):
        handle.write(
            f"ATOM  {k:5d}  CA  ALA A{k:4d}    "
            f"{p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}"
            "  1.00  0.00           C\n"
        )
    handle.write("ENDMDL\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged-csv", required=True, type=Path)
    ap.add_argument("--v9-combined-pdb", required=True, type=Path)
    ap.add_argument("--v9-manifest", required=True, type=Path)
    ap.add_argument("--v9-train-idx", required=True, type=Path)
    ap.add_argument("--v9-test-idx", required=True, type=Path)
    ap.add_argument("--pdb-dir", required=True, type=Path)
    ap.add_argument("--ref-pdb", required=True, type=Path)
    ap.add_argument("--ref-chain", default="C")
    ap.add_argument("--ref-dfg", type=int, default=594)
    ap.add_argument("--ref-ape", type=int, default=623)
    ap.add_argument("--flank", type=int, default=40)
    ap.add_argument("--min-flank-frac", type=float, default=0.7)
    ap.add_argument("--min-loop-frac", type=float, default=0.7)
    ap.add_argument("--flank-rmsd-max", type=float, default=8.0)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=25)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- reference flank ----
    ref_bb = read_backbone(args.ref_pdb, args.ref_chain)
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

    # ---- load v9 in-dist Cα models ----
    v9_models = load_v9_indist_models(args.v9_combined_pdb,
                                       args.v9_manifest)
    v9_manifest = pd.read_csv(args.v9_manifest, keep_default_na=False)
    v9_manifest["chain_key"] = v9_manifest["chain_key"].str.upper()
    print(f"v9 in-dist Cα models loaded: {len(v9_models)}")

    # ---- load merged set ----
    merged = pd.read_csv(args.merged_csv, keep_default_na=False)
    merged["chain_key"] = merged["chain_key"].str.upper()
    print(f"merged kinome: {len(merged)} chains "
          f"({(merged['source'] == 'v9_indist').sum()} v9_indist + "
          f"{(merged['source'] != 'v9_indist').sum()} new OOD)")

    out_pdb = args.out_dir / "combined_v91_ca.pdb"
    out_manifest_rows = []
    failures = []
    fh = out_pdb.open("w")
    model_idx = 0

    # Pass 1: write v9 in-dist models in their existing order, with
    # original Cα coords.
    v9_keys_in_order = v9_manifest["chain_key"].tolist()
    v9_keyset = set(v9_keys_in_order)
    print("writing v9 in-dist chains ...")
    for ck in v9_keys_in_order:
        coords = v9_models.get(ck)
        if coords is None:
            failures.append({"chain_key": ck, "status": "missing_in_v9_pdb"})
            continue
        write_ca_pdb(fh, coords, model_idx)
        # Lookup metadata from merged (or v9_manifest for in-dist details)
        row_v9 = v9_manifest[v9_manifest["chain_key"] == ck].iloc[0]
        out_manifest_rows.append({
            "chain_key":   ck,
            "pdb":         ck[:4],
            "chain":       ck[4:],
            "gene":        row_v9.get("gene", ""),
            "group":       row_v9.get("group", ""),
            "dfg_spatial": row_v9.get("dfg_spatial", ""),
            "dihedral":    row_v9.get("dihedral", ""),
            "ligand_type": row_v9.get("ligand_type", ""),
            "anchor_motif": "",
            "loop_length": int(row_v9.get("expected_loop", 0)),
            "flank_rmsd":  float(row_v9.get("flank_rmsd", 0)),
            "source":      "v9_indist",
            "model_idx":   model_idx,
        })
        model_idx += 1

    # Pass 2: re-extract new OOD chains
    new_chains = merged[merged["source"] != "v9_indist"]
    print(f"extracting {len(new_chains)} new OOD chains ...")
    n_ok = 0
    for i, (_, row) in enumerate(new_chains.iterrows()):
        ck = str(row["chain_key"]).upper()
        if ck in v9_keyset:
            # safety: skip duplicates
            continue
        res, status = extract_new_chain(ck, args.pdb_dir, ref_specs,
                                          ref_flank, args)
        if res is None:
            failures.append({"chain_key": ck, "status": status})
            continue
        write_ca_pdb(fh, res["coords"], model_idx)
        out_manifest_rows.append({
            "chain_key":   ck,
            "pdb":         ck[:4],
            "chain":       ck[4:],
            "gene":        row.get("gene", ""),
            "group":       row.get("group", ""),
            "dfg_spatial": row.get("dfg_spatial", ""),
            "dihedral":    row.get("dihedral", ""),
            "ligand_type": row.get("ligand_type", ""),
            "anchor_motif": res["anchor_motif"],
            "loop_length": res["loop_length"],
            "flank_rmsd":  res["flank_rmsd"],
            "source":      str(row.get("source", "new_ood")),
            "model_idx":   model_idx,
        })
        model_idx += 1
        n_ok += 1
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(new_chains)}, ok={n_ok}, fail={len(failures)}")
    fh.write("END\n"); fh.close()
    print(f"\nWrote {out_pdb} with {model_idx} MODELs "
          f"(of {len(merged)} chains, {len(failures)} failures)")

    manifest_df = pd.DataFrame(out_manifest_rows)
    manifest_df.to_csv(args.out_dir / "manifest_v91.csv", index=False)
    pd.DataFrame(failures).to_csv(
        args.out_dir / "failures_v91.csv", index=False)

    # ---- new train/test split ----
    n_v9 = (manifest_df["source"] == "v9_indist").sum()
    old_train_idx = np.loadtxt(args.v9_train_idx, dtype=int)
    old_test_idx = np.loadtxt(args.v9_test_idx, dtype=int)
    # v9-indist chains keep their old indices
    # New chains: append at the end and assign 90/10
    n_total = len(manifest_df)
    new_indices = np.arange(n_v9, n_total)
    rng = np.random.default_rng(args.seed)
    n_new_test = max(1, int(len(new_indices) * 0.10))
    new_test_choice = rng.choice(new_indices, size=n_new_test, replace=False)
    new_train_choice = np.setdiff1d(new_indices, new_test_choice)
    train_idx = np.concatenate([old_train_idx, new_train_choice])
    test_idx = np.concatenate([old_test_idx, new_test_choice])
    np.savetxt(args.out_dir / "train_idx.txt", train_idx, fmt="%d")
    np.savetxt(args.out_dir / "test_idx.txt", test_idx, fmt="%d")
    print(f"Train/test split: {len(train_idx)}/{len(test_idx)} "
          f"({100 * len(test_idx) // n_total}% test)")


if __name__ == "__main__":
    main()
