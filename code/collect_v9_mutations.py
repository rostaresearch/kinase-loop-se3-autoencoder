"""Extract mutation annotations for every v9 chain from PDB headers.

This script makes NO literature claims. It only parses what is written
in the PDB file's header records:

  - COMPND record: looks for MUTATION: YES and MUTANT:/MUT: tags
  - TITLE record: looks for explicit mutation strings like "V600E"
  - REMARK 999: free-text mutation notes
  - SEQADV records: official residue-level mutation annotations

The resulting CSV records every (chain, annotation source, text fragment,
extracted single-mutation hits) so any downstream literature lookup is
auditable: the user can verify each call against the original PDB header.

Output: ``v9_chain_mutations.csv`` with columns
  chain_key, pdb, chain, gene, group, dfg_spatial, dihedral, ligand_type,
  n_seqadv_mutations,        # number of SEQADV "MUTATION" lines
  seqadv_mutations,          # comma-separated list of mutation strings from SEQADV
  title_mutation_hits,       # mutation-like tokens found in TITLE
  compnd_mutation,           # COMPND-record MUTATION value (YES/NO/none)
  remark_999_mutation_lines, # raw REMARK 999 lines mentioning MUTATION
  has_any_mutation_annotation
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


# Regex for "AaaNNN..." mutation token, e.g. V600E, T315I, T790M, L858R.
# Pattern: 1-letter aa, 2-5 digit residue number, 1-letter aa.
MUTATION_TOKEN = re.compile(r"\b([ACDEFGHIKLMNPQRSTVWY])(\d{2,5})([ACDEFGHIKLMNPQRSTVWY])\b")


def parse_pdb_header(pdb_path: Path, chain_id: str) -> dict:
    out = {
        "compnd_mutation": "none",
        "title_mutation_hits": [],
        "remark_999_mutation_lines": [],
        "seqadv_mutations": [],
    }
    if not pdb_path.exists():
        return out

    title_lines, compnd_lines, remark_lines, seqadv_lines = [], [], [], []
    with pdb_path.open() as f:
        for line in f:
            tag = line[:6].rstrip()
            if tag == "ATOM" or tag == "HETATM":
                break  # done with header
            if tag == "TITLE":
                title_lines.append(line[10:].strip())
            elif tag == "COMPND":
                compnd_lines.append(line[10:].strip())
            elif tag == "REMARK":
                # Only REMARK 999 (free-text mutations).
                if line[7:10].strip() == "999":
                    remark_lines.append(line[11:].strip())
            elif tag == "SEQADV":
                seqadv_lines.append(line.rstrip())

    title_text = " ".join(title_lines).upper()
    compnd_text = " ".join(compnd_lines).upper()
    remark_text = " ".join(remark_lines)

    # COMPND MUTATION: YES/NO
    m = re.search(r"MUTATION\s*:?\s*(YES|NO)", compnd_text)
    if m:
        out["compnd_mutation"] = m.group(1)

    # Mutation tokens in TITLE.
    title_hits = MUTATION_TOKEN.findall(title_text)
    out["title_mutation_hits"] = [
        f"{a1}{num}{a2}" for a1, num, a2 in title_hits
    ]

    # REMARK 999 lines that mention MUTATION/MUTANT.
    rem_relevant = [
        line for line in remark_lines
        if re.search(r"\bMUTATION\b|\bMUTANT\b", line, re.I)
    ]
    out["remark_999_mutation_lines"] = rem_relevant

    # SEQADV: chain-specific mutation records.
    # Standard SEQADV format:
    # SEQADV ID  RESN CHN RESID ICODE DBSWS DBACC DBID DBRES DBSEQ CONFLICT_TYPE
    # We filter chain-matches with CONFLICT_TYPE = "MUTATION".
    sm = []
    for line in seqadv_lines:
        # chain is at column 16 (PDB v3 convention)
        if len(line) < 17:
            continue
        chn = line[16]
        if chn != chain_id:
            continue
        upper = line.upper()
        if "MUTATION" in upper or "ENGINEERED" in upper:
            # extract residue name+number+sourceres if available
            # SEQADV layout: cols 7-11 = ID; 12-14 = RESN (3-letter PDB aa);
            # 16 = chain; 18-22 = resseq; 24-26 = DBRES (3-letter source aa).
            try:
                resn = line[12:15].strip()
                resseq = line[18:22].strip()
                dbres = line[39:42].strip() if len(line) >= 42 else ""
                sm.append(f"{dbres}->{resn}@{resseq}" if dbres else
                          f"?->{resn}@{resseq}")
            except Exception:
                sm.append(line[12:].strip())
    out["seqadv_mutations"] = sm
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-csv", required=True, type=Path)
    ap.add_argument("--full-pdb-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest_csv, keep_default_na=False)

    rows = []
    for i, r in manifest.iterrows():
        if i % 200 == 0:
            print(f"  parsing chain {i}/{len(manifest)}")
        pdb = str(r["pdb"]); chain = str(r["chain"])
        info = parse_pdb_header(
            args.full_pdb_dir / f"{pdb}.pdb", chain
        )
        rows.append({
            "chain_key": str(r["chain_key"]).upper(),
            "pdb": pdb, "chain": chain,
            "gene": r.get("gene", ""),
            "group": r.get("group", ""),
            "dfg_spatial": r.get("dfg_spatial", ""),
            "dihedral": r.get("dihedral", ""),
            "ligand_type": r.get("ligand_type", ""),
            "compnd_mutation": info["compnd_mutation"],
            "n_seqadv_mutations": len(info["seqadv_mutations"]),
            "seqadv_mutations":
                "; ".join(info["seqadv_mutations"]),
            "title_mutation_hits":
                ",".join(info["title_mutation_hits"]),
            "n_remark_999_mutation_lines":
                len(info["remark_999_mutation_lines"]),
            "remark_999_mutation_lines":
                " || ".join(info["remark_999_mutation_lines"][:5]),
            "has_any_mutation_annotation": bool(
                info["seqadv_mutations"]
                or info["title_mutation_hits"]
                or info["remark_999_mutation_lines"]
                or info["compnd_mutation"] == "YES"
            ),
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}  ({len(df)} chains)")
    print(f"  with any mutation annotation: "
          f"{df['has_any_mutation_annotation'].sum()}")
    print(f"  with SEQADV mutation record:  "
          f"{(df['n_seqadv_mutations']>0).sum()}")
    print(f"  with TITLE mutation token:    "
          f"{(df['title_mutation_hits']!='').sum()}")
    print(f"  COMPND MUTATION=YES:          "
          f"{(df['compnd_mutation']=='YES').sum()}")

    # Per-gene summary: any kinase with both WT and mutant structures?
    grp = df.groupby("gene").agg(
        n_chains=("chain_key", "size"),
        n_wt=("has_any_mutation_annotation", lambda x: (~x).sum()),
        n_mutant=("has_any_mutation_annotation", "sum"),
    ).reset_index()
    grp = grp[(grp["n_wt"] > 0) & (grp["n_mutant"] > 0)]
    grp = grp.sort_values("n_mutant", ascending=False)
    grp.to_csv(args.out.parent / "v9_genes_with_wt_and_mutant.csv",
               index=False)
    print(f"\nGenes with BOTH WT and mutant chains: {len(grp)}")
    print(grp.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
