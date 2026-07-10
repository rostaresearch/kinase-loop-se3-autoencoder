"""Build the v9 dataset: Cα-only spline-fit, length 27 (BRAF-matched).

This is the simplification you asked for: we no longer spline-fit the
non-Cα backbone atoms (which had no chemistry constraints and could
distort N–CA, CA–C, C=O bond geometry).  Only Cα is splined; that
preserves the variable-loop-length normalisation across kinases.

The 6 chemically-diagnostic DFG/APE anchor residues (D, F, G, A, P, E)
are written to a separate sidecar PDB with their raw, untouched
N/CA/C/O/CB backbone (post-Kabsch but no spline), in case we want
sidechain-orientation analysis downstream.  These are NOT used to
train the autoencoder.

Spline math is an exact port of meyresearch/BRAF's
``workflow/fitting_class.py`` (``Fitting._fit_cubic_interpolation`` and
``_calculate_arc_length_parameterization``):

  fits[d] = interp1d(np.arange(n), coords[:, j], kind='cubic')   # res-index param
  X       = np.arange(0, n-1, 0.1)
  Y       = sqrt(sum(grad(fits[d](X))**2 for d in xyz))
  L       = trapz(Y, X)
  Li      = np.linspace(0, L, Nnew)
  pt[i]   = argmin |flen - Li[i]|
  new_xyz = fits[d](X[pt[i]])

Pipeline per chain:
  1. read full PDB chain; pull N/Cα/C/O/Cβ in residues DFG-D … APE-E,
     plus the 80 flank CAs (40 before DFG-D, 40 after APE-E)
  2. Kabsch-align the flank Cα atoms to BRAF 6UAN chain C
  3. apply rigid transform to ALL atoms (loop + anchors)
  4. extract the loop CA trace (length K_present residues, K_present ≤ K)
  5. cubic spline fit on residue index, arc-length resample to N_loop_points
  6. emit two PDBs:
       combined_v9_ca.pdb   — 27 CAs per chain, single ALA dummies (AE input)
       combined_v9_anchors.pdb — 6 anchor residues × 5 atoms per chain (no spline)
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm

warnings.filterwarnings("ignore")

BACKBONE = ("N", "CA", "C", "CB", "O")     # molearn order
ANCHOR_OFFSETS = {                          # offsets from DFG/APE motif starts
    "DFG-D": ("dfg", 0),
    "DFG-F": ("dfg", 1),
    "DFG-G": ("dfg", 2),
    "APE-A": ("ape", -2),
    "APE-P": ("ape", -1),
    "APE-E": ("ape",  0),
}


def read_backbone(pdb_path: Path, chain_id: str
                  ) -> dict[int, dict[str, np.ndarray]] | None:
    """``{resi: {atom_name: xyz, '__name__': resname}}`` for ATOM lines."""
    out: dict[int, dict[str, np.ndarray]] = {}
    resnames: dict[int, str] = {}
    with pdb_path.open() as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[21] != chain_id:
                continue
            atom = line[12:16].strip()
            if atom not in BACKBONE:
                continue
            try:
                resi = int(line[22:26])
            except ValueError:
                continue
            if resi not in out:
                out[resi] = {}
                resnames[resi] = line[17:20].strip()
            if atom in out[resi]:
                continue  # first altloc only
            out[resi][atom] = np.array([float(line[30:38]),
                                         float(line[38:46]),
                                         float(line[46:54])],
                                        dtype=np.float32)
    for r in out:
        out[r]["__name__"] = resnames[r]   # type: ignore
    return out if out else None


def virtual_cb(n, ca, c):
    """1.5 Å tetrahedral bisector — for Gly only."""
    bisector = (ca - n) + (ca - c)
    bisector /= (np.linalg.norm(bisector) + 1e-9)
    return ca + 1.5 * bisector


def kabsch(mob: np.ndarray, ref: np.ndarray):
    """Rigid R, t mapping mob → ref; returns (R, t, rmsd)."""
    mc = mob.mean(axis=0); rc = ref.mean(axis=0)
    A = mob - mc; B = ref - rc
    H = A.T @ B
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    A_rot = A @ R.T
    rmsd = float(np.sqrt(((A_rot - B) ** 2).sum(axis=1).mean()))
    return R, rc - mc @ R.T, rmsd


# -------------------------------------------------------------- spline


def spline_ca_arclen(ca_coords: np.ndarray, n_out: int,
                     fine_step: float = 0.1
                     ) -> np.ndarray | None:
    """Port of meyresearch ``Fitting._fit_cubic_interpolation`` +
    ``_calculate_arc_length_parameterization``.

    ca_coords : (K, 3) Cα positions (NaN rows must already be dropped).
    n_out     : number of output points (27 for BRAF).
    fine_step : resolution of the residue-index grid used to compute
                arc length (smaller = more accurate, slower).
    Returns (n_out, 3) Cα positions equally spaced along arc length,
    or None on failure.
    """
    n = ca_coords.shape[0]
    if n < 4:
        return None
    # Cubic on residue-index parameter.
    fits = [interp1d(np.arange(n), ca_coords[:, j],
                     kind='cubic', fill_value='extrapolate')
            for j in range(3)]
    # Walk a fine grid to compute speed → arc length.
    X = np.arange(0.0, n - 1, fine_step)
    if X.size < 2:
        return None
    grads = [np.gradient(fits[j](X)) for j in range(3)]
    speed = np.sqrt(grads[0]**2 + grads[1]**2 + grads[2]**2)
    L = np.trapz(speed, X)
    if L < 1e-3:
        return None
    Li = np.linspace(0.0, L, n_out)
    # Cumulative arc length at each fine-grid point (start at 0).
    flen = np.concatenate([
        [0.0],
        np.array([np.trapz(speed[:i + 1], X[:i + 1])
                  for i in range(1, len(X))]),
    ])
    pt_idx = np.array([int(np.argmin(np.abs(flen - Li[i])))
                       for i in range(n_out)])
    Xq = X[pt_idx]
    out = np.stack([fits[j](Xq) for j in range(3)], axis=1)
    return out.astype(np.float32)


# -------------------------------------------------------------- per-chain


def process_chain(pdb_path: Path, chain_id: str,
                  dfg_resi: int, ape_resi: int,
                  ref_flank_coords: np.ndarray,
                  ref_flank_specs: list[tuple[str, int]],
                  ape_offset_to_e: int,
                  flank: int, min_flank_frac: float,
                  min_loop_frac: float, flank_rmsd_max: float,
                  n_loop_points: int):
    bb = read_backbone(pdb_path, chain_id)
    if bb is None:
        return {"status": "no_chain"}

    # ----- flank Cα Kabsch alignment -----
    flank_mob, flank_ref = [], []
    for (anchor, off), ref_xyz in zip(ref_flank_specs, ref_flank_coords):
        if anchor == "dfg":
            target = dfg_resi + off
        elif anchor == "ape":
            target = ape_resi + off
        else:
            raise ValueError(anchor)
        if target in bb and "CA" in bb[target]:
            flank_mob.append(bb[target]["CA"])
            flank_ref.append(ref_xyz)
    if len(flank_mob) < min_flank_frac * len(ref_flank_specs):
        return {"status": "underresolved_flank",
                "n_flank_present": len(flank_mob)}
    mob = np.array(flank_mob, dtype=np.float32)
    ref = np.array(flank_ref, dtype=np.float32)
    R, t, rmsd = kabsch(mob, ref)
    if rmsd > flank_rmsd_max:
        return {"status": "flank_rmsd_too_high", "rmsd": rmsd}
    mc = mob.mean(axis=0); rc = ref.mean(axis=0)

    def transform(xyz: np.ndarray) -> np.ndarray:
        return (xyz - mc) @ R.T + rc

    # ----- loop Cα (length K residues, may have gaps) -----
    loop_resis = list(range(dfg_resi, ape_resi + 1))
    K = len(loop_resis)
    ca_present = []
    for r in loop_resis:
        if r in bb and "CA" in bb[r]:
            ca_present.append(transform(bb[r]["CA"]))
    if len(ca_present) < min_loop_frac * K:
        return {"status": "underresolved_loop",
                "n_loop_present": len(ca_present),
                "expected": K}
    ca_arr = np.array(ca_present, dtype=np.float32)

    # ----- spline-fit Cα to N output points -----
    ca_resampled = spline_ca_arclen(ca_arr, n_loop_points)
    if ca_resampled is None:
        return {"status": "ca_spline_failed"}

    # ----- anchor residues (raw, post-Kabsch, no spline) -----
    anchors: dict[str, dict[str, np.ndarray]] = {}
    for name, (kind, off) in ANCHOR_OFFSETS.items():
        if kind == "dfg":
            r_resi = dfg_resi + off
        else:
            # APE motif starts at ape_resi - ape_offset_to_e (= APE-A).
            ape_a = ape_resi - ape_offset_to_e
            r_resi = ape_a + (off + ape_offset_to_e)
            # i.e. APE-A at off=-2 → ape_a; APE-P at off=-1 → ape_a+1;
            # APE-E at off=0  → ape_a+2 = ape_resi.
        if r_resi not in bb:
            anchors[name] = {}
            continue
        res_data = bb[r_resi]
        is_gly = (res_data.get("__name__") == "GLY")
        atoms_xyz: dict[str, np.ndarray] = {}
        for a in BACKBONE:
            if a in res_data:
                atoms_xyz[a] = transform(res_data[a])
            elif a == "CB" and is_gly:
                if all(x in res_data for x in ("N", "CA", "C")):
                    atoms_xyz[a] = transform(virtual_cb(
                        res_data["N"], res_data["CA"], res_data["C"]
                    ))
        anchors[name] = atoms_xyz

    return {"status": "ok",
            "ca": ca_resampled,
            "anchors": anchors,
            "flank_rmsd": rmsd,
            "n_loop_present": len(ca_present),
            "expected": K}


# -------------------------------------------------------------- writers


def write_ca_pdb(handle, coords: np.ndarray, model_idx: int):
    """Append one MODEL block with n CAs to combined PDB."""
    handle.write(f"MODEL {model_idx}\n")
    for k, p in enumerate(coords, start=1):
        handle.write(
            f"ATOM  {k:5d}  CA  ALA A{k:4d}    "
            f"{p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}"
            "  1.00  0.00           C\n"
        )
    handle.write("ENDMDL\n")


def write_anchor_pdb(handle, anchors: dict[str, dict[str, np.ndarray]],
                     model_idx: int):
    """Append one MODEL block with the 6 anchor residues × up to 5 atoms.

    Residues are written in fixed order DFG-D, DFG-F, DFG-G, APE-A,
    APE-P, APE-E.  Atom order: N, CA, C, CB, O.  Missing atoms are
    skipped (no NaN padding — the AE never sees this file).
    """
    residue_names_three = {
        "DFG-D": "ASP", "DFG-F": "PHE", "DFG-G": "GLY",
        "APE-A": "ALA", "APE-P": "PRO", "APE-E": "GLU",
    }
    handle.write(f"MODEL {model_idx}\n")
    serial = 1
    for resi_idx, anchor_name in enumerate(
            ["DFG-D", "DFG-F", "DFG-G", "APE-A", "APE-P", "APE-E"],
            start=1):
        atoms = anchors.get(anchor_name, {})
        if not atoms:
            continue
        rname = residue_names_three[anchor_name]
        for a in BACKBONE:
            if a not in atoms:
                continue
            p = atoms[a]
            handle.write(
                f"ATOM  {serial:5d}  {a:<3s} {rname} A{resi_idx:4d}    "
                f"{p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}"
                f"  1.00  0.00           {a[0]}\n"
            )
            serial += 1
    handle.write("ENDMDL\n")


# -------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assignments-csv", required=True)
    ap.add_argument("--full-pdb-dir", required=True)
    ap.add_argument("--ref-pdb", required=True)
    ap.add_argument("--ref-chain", required=True)
    ap.add_argument("--ref-dfg", type=int, default=594)
    ap.add_argument("--ref-ape", type=int, default=623,
                    help="Reference APE-E residue (NOT motif start).")
    ap.add_argument("--ape-offset-to-e", type=int, default=2)
    ap.add_argument("--kincore-fasta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-loop-points", type=int, default=27,
                    help="BRAF activation loop length.")
    ap.add_argument("--flank", type=int, default=40)
    ap.add_argument("--min-flank-frac", type=float, default=0.7)
    ap.add_argument("--min-loop-frac", type=float, default=0.7)
    ap.add_argument("--flank-rmsd-max", type=float, default=5.0,
                    help="Flank Cα RMSD cutoff in Å.  meyresearch use "
                         "3 Å on the terminal CA only; we keep 5 Å on "
                         "the 80-flank because it averages out single "
                         "residue noise.")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # ----- reference flank -----
    ref_bb = read_backbone(Path(args.ref_pdb), args.ref_chain)
    if ref_bb is None:
        raise SystemExit(f"Could not read {args.ref_pdb}")
    ref_specs: list[tuple[str, int]] = []
    ref_flank = []
    for r in range(args.ref_dfg - args.flank, args.ref_dfg):
        if r in ref_bb and "CA" in ref_bb[r]:
            ref_specs.append(("dfg", r - args.ref_dfg))
            ref_flank.append(ref_bb[r]["CA"])
    for r in range(args.ref_ape + 1, args.ref_ape + args.flank + 1):
        if r in ref_bb and "CA" in ref_bb[r]:
            ref_specs.append(("ape", r - args.ref_ape))
            ref_flank.append(ref_bb[r]["CA"])
    ref_flank = np.array(ref_flank, dtype=np.float32)
    print(f"Reference flank atoms: {len(ref_specs)}/{2 * args.flank}")

    # ----- Kincore labels for manifest -----
    klab = {}
    with open(args.kincore_fasta) as f:
        for line in f:
            if not line.startswith(">"):
                continue
            parts = line.rstrip().split("\t")
            ident = parts[0][1:]
            if ident.startswith("AF-"):
                continue
            klab[ident.upper()] = {
                "gene":        parts[1] if len(parts) > 1 else "",
                "group":       parts[3] if len(parts) > 3 else "",
                "dfg_spatial": parts[4] if len(parts) > 4 else "",
                "dihedral":    parts[5] if len(parts) > 5 else "",
                "ligand_type": parts[7] if len(parts) > 7 else "",
            }

    df = pd.read_csv(args.assignments_csv)
    df = df[df["status"] == "selected"]
    if args.limit:
        df = df.head(args.limit)
    print(f"Chains to process: {len(df)}")

    pdb_dir = Path(args.full_pdb_dir)
    manifest_rows = []
    failure_rows = []

    combined_ca = (out / "combined_v9_ca.pdb").open("w")
    combined_anchors = (out / "combined_v9_anchors.pdb").open("w")
    model_idx = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        chain_key = str(row["chain_key"]).upper()
        pdb_id = chain_key[:4]
        chain_id = chain_key[4:]
        try:
            dfg = int(row["selected_dfg_resi"])
            ape_motif_start = int(row["selected_ape_resi"])
            ape = ape_motif_start + args.ape_offset_to_e
        except (ValueError, TypeError, KeyError):
            failure_rows.append({"chain_key": chain_key,
                                 "status": "bad_dfg_ape"})
            continue
        pdb_path = pdb_dir / f"{pdb_id}.pdb"
        if not pdb_path.exists():
            failure_rows.append({"chain_key": chain_key,
                                 "status": "no_pdb_file"})
            continue
        try:
            r = process_chain(
                pdb_path, chain_id, dfg, ape,
                ref_flank, ref_specs,
                ape_offset_to_e=args.ape_offset_to_e,
                flank=args.flank,
                min_flank_frac=args.min_flank_frac,
                min_loop_frac=args.min_loop_frac,
                flank_rmsd_max=args.flank_rmsd_max,
                n_loop_points=args.n_loop_points,
            )
        except Exception as e:
            failure_rows.append({"chain_key": chain_key,
                                 "status": f"exception:{e}"})
            continue
        if r["status"] != "ok":
            row_out = {"chain_key": chain_key, **r}
            for kkey in ("ca", "anchors"):
                row_out.pop(kkey, None)
            failure_rows.append(row_out)
            continue
        write_ca_pdb(combined_ca, r["ca"], model_idx)
        write_anchor_pdb(combined_anchors, r["anchors"], model_idx)
        model_idx += 1
        info = klab.get(chain_key, {})
        anchor_atom_counts = {
            f"anchor_{name}_atoms": len(r["anchors"].get(name, {}))
            for name in ANCHOR_OFFSETS
        }
        manifest_rows.append({
            "chain_key": chain_key, "pdb": pdb_id, "chain": chain_id,
            "dfg_resi": dfg,
            "ape_motif_start_resi": ape_motif_start,
            "ape_resi": ape,
            "flank_rmsd": r["flank_rmsd"],
            "n_loop_present": r["n_loop_present"],
            "expected_loop": r["expected"],
            **info,
            **anchor_atom_counts,
        })
    combined_ca.write("END\n"); combined_ca.close()
    combined_anchors.write("END\n"); combined_anchors.close()

    pd.DataFrame(manifest_rows).to_csv(out / "manifest_v9.csv",
                                       index=False)
    pd.DataFrame(failure_rows).to_csv(out / "failures_v9.csv",
                                      index=False)
    print(f"\nWrote {len(manifest_rows)} models to {out}/combined_v9_ca.pdb")
    print(f"Failures: {len(failure_rows)}")
    if manifest_rows:
        from collections import Counter
        print("DFG spatial:",
              Counter([r.get("dfg_spatial", "") for r in manifest_rows]))
        print("Dihedral:",
              Counter([r.get("dihedral", "") for r in manifest_rows]))
        n_anchor_complete = sum(
            1 for r in manifest_rows
            if all(r[f"anchor_{a}_atoms"] >= 4 for a in ANCHOR_OFFSETS)
        )
        print(f"Chains with ≥4 atoms in all 6 anchors: "
              f"{n_anchor_complete}/{len(manifest_rows)}")


if __name__ == "__main__":
    main()
