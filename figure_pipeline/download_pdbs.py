"""Raw-data acquisition: download exactly the PDB structures this pipeline needs.

The 6,531-chain dataset references **4,533 unique PDB entries**. You do NOT need the
full ~8,200-entry InterPro sweep to reproduce the results — derive the list from the
manifest (preferred, exact) or from the original InterPro IPR011009 list.

Usage
-----
  # exact set the pipeline uses (recommended):
  python download_pdbs.py --manifest-csv manifest_v91.csv --out-dir PDBs

  # or the full original InterPro IPR011009 sweep (superset, ~8,223):
  python download_pdbs.py --ipr-tsv structure-matching-IPR011009.tsv --out-dir PDBs

Notes
-----
* Source: https://files.rcsb.org/download/{ID}.pdb  (public, no auth)
* Idempotent: existing, non-empty files are skipped, so it is safe to re-run/resume.
* --check-only reports what is missing without downloading.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request, urllib.error

URL = "https://files.rcsb.org/download/{}.pdb"


def ids_from_manifest(p: Path) -> list[str]:
    import pandas as pd
    m = pd.read_csv(p, keep_default_na=False)
    col = "pdb" if "pdb" in m.columns else "chain_key"
    if col == "pdb":
        ids = m["pdb"].astype(str).str.upper()
    else:  # chain_key like 1A9UA -> first 4 chars
        ids = m["chain_key"].astype(str).str.upper().str[:4]
    return sorted(set(i for i in ids if len(i) == 4))


def ids_from_ipr(p: Path) -> list[str]:
    import pandas as pd
    t = pd.read_csv(p, sep="\t", keep_default_na=False)
    return sorted({str(a).upper() for a in t["Accession"] if len(str(a)) == 4})


def fetch(pdb_id: str, out_dir: Path, retries: int = 3) -> tuple[str, str]:
    dest = out_dir / f"{pdb_id}.pdb"
    if dest.exists() and dest.stat().st_size > 0:
        return pdb_id, "skip"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(URL.format(pdb_id), timeout=30) as r:
                data = r.read()
            if not data:
                raise ValueError("empty")
            dest.write_bytes(data)
            return pdb_id, "ok"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return pdb_id, "404"          # obsolete/withdrawn entry
            time.sleep(1 + attempt)
        except Exception:
            time.sleep(1 + attempt)
    return pdb_id, "fail"


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest-csv", type=Path, help="manifest_v91.csv (exact set: 4,533 IDs)")
    src.add_argument("--ipr-tsv", type=Path, help="structure-matching-IPR011009.tsv (superset)")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()

    ids = ids_from_manifest(a.manifest_csv) if a.manifest_csv else ids_from_ipr(a.ipr_tsv)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    have = {p.stem.upper() for p in a.out_dir.glob("*.pdb") if p.stat().st_size > 0}
    todo = [i for i in ids if i not in have]
    print(f"required: {len(ids)} | already present: {len(ids)-len(todo)} | to download: {len(todo)}")
    if a.check_only:
        print("missing:", " ".join(todo[:50]), "..." if len(todo) > 50 else "")
        return

    counts = {"ok": 0, "skip": 0, "404": 0, "fail": 0}
    gone = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(fetch, i, a.out_dir): i for i in todo}
        for n, f in enumerate(as_completed(futs), 1):
            pid, st = f.result()
            counts[st] += 1
            if st in ("404", "fail"):
                gone.append(f"{pid}:{st}")
            if n % 200 == 0:
                print(f"  {n}/{len(todo)}  {counts}")
    print("done:", counts)
    if gone:
        (a.out_dir / "_download_problems.txt").write_text("\n".join(gone))
        print(f"{len(gone)} problem entries -> {a.out_dir/'_download_problems.txt'}")
        print("(404 = obsolete/superseded PDB entry; re-check against the manifest)")
    still = [i for i in ids if not (a.out_dir / f"{i}.pdb").exists()]
    print(f"FINAL: {len(ids)-len(still)}/{len(ids)} present" + (f" | STILL MISSING {len(still)}" if still else " | complete"))
    sys.exit(1 if still else 0)


if __name__ == "__main__":
    main()
