"""Enumerate all PDB-annotated mutations in v9 and run M1 + significance on each.

Reads v9_chain_mutations.csv (PDB-header-extracted annotations) and the
v9 latent CSV. For every (gene, mutation) pair with n_mutant >= 2 chains
in v9:

  - normalises mutation tokens from title_mutation_hits, seqadv_mutations,
    and remark_999_mutation_lines into "X{N}Y" form
  - M1: verifies the WT residue at position N matches the UniProt canonical
    sequence of that gene's reviewed human kinase
  - Significance: permutation test + Mahalanobis distance + bootstrap CI
    on the WT vs mutant latent centroid distance

Outputs a single table v9_mutation_validation_skeleton.csv with all of:
  gene, mutation, n_wt_chains, n_mut_chains, n_mut_pdbs,
  m1_uniprot_acc, m1_canonical_aa, m1_status,
  wt_z0_mean, wt_z1_mean, mut_z0_mean, mut_z1_mean,
  delta_latent, delta_in_wt_sigmas, mahalanobis_sigma,
  perm_pvalue, boot_delta_median, boot_delta_2.5pct, boot_delta_97.5pct,
  significant_perm_p<0.05, significant_mahal_>chi2(0.99)
  -- PLACEHOLDER columns for OncoKB join:
  oncokb_oncogenic, oncokb_level, oncokb_drug_context

Plus an OncoKB-join-ready key column gene_mut_token ("BRAF:V600E").

When the OncoKB TSVs are dropped at manuscript_draft/data/oncokb_static/,
a downstream join_oncokb_to_skeleton.py will fill in the placeholder
columns by matching on gene_mut_token.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

MUT_REGEX = re.compile(r"\b([ACDEFGHIKLMNPQRSTVWY])(\d{2,5})([ACDEFGHIKLMNPQRSTVWY])\b")
SEQADV_REGEX = re.compile(r"([A-Z]{3})->([A-Z]{3})@(\d+)")


# ---------------------------------------------------------------- mutation extraction


def extract_mutations_for_chain(row: pd.Series) -> set[str]:
    out: set[str] = set()

    title = str(row.get("title_mutation_hits", "") or "")
    for tok in MUT_REGEX.findall(title.upper()):
        out.add(f"{tok[0]}{tok[1]}{tok[2]}")

    seqadv = str(row.get("seqadv_mutations", "") or "")
    for m in SEQADV_REGEX.findall(seqadv.upper()):
        wt3, mt3, pos = m
        wt1 = THREE_TO_ONE.get(wt3); mt1 = THREE_TO_ONE.get(mt3)
        if wt1 and mt1 and wt1 != mt1:
            out.add(f"{wt1}{pos}{mt1}")

    rem = str(row.get("remark_999_mutation_lines", "") or "")
    for tok in MUT_REGEX.findall(rem.upper()):
        out.add(f"{tok[0]}{tok[1]}{tok[2]}")

    return out


# ---------------------------------------------------------------- UniProt M1


UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_FASTA = "https://rest.uniprot.org/uniprotkb/{acc}.fasta"


def http_text(url, params=None, retries=3, sleep=0.5):
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    last = None
    for _ in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as f:
                return f.read().decode("utf-8")
        except Exception as e:
            last = e
            time.sleep(sleep)
    raise RuntimeError(f"HTTP failed: {url}  ({last})")


def http_json(url, params=None, retries=3, sleep=0.5):
    return json.loads(http_text(url, params=params, retries=retries, sleep=sleep))


def fetch_uniprot_seq(gene: str, cache: dict) -> tuple[str | None, str | None]:
    if gene in cache:
        return cache[gene]
    try:
        data = http_json(UNIPROT_SEARCH, params={
            "query": f"gene_exact:{gene} AND organism_id:9606 AND reviewed:true",
            "fields": "accession,id,gene_names",
            "format": "json", "size": 5})
        if not data.get("results"):
            cache[gene] = (None, None)
            return None, None
        acc = data["results"][0]["primaryAccession"]
        for r in data["results"]:
            for gn in r.get("genes", []):
                primary = (gn.get("geneName") or {}).get("value")
                if primary and primary.upper() == gene.upper():
                    acc = r["primaryAccession"]
                    break
        fasta = http_text(UNIPROT_FASTA.format(acc=acc))
        seq = "".join(L.strip() for L in fasta.splitlines() if not L.startswith(">"))
        cache[gene] = (acc, seq)
        time.sleep(0.2)
        return acc, seq
    except Exception as e:
        print(f"  UniProt fetch failed for {gene}: {e}")
        cache[gene] = (None, None)
        return None, None


def m1_check(gene: str, mut: str, cache: dict) -> dict:
    if len(mut) < 3:
        return {"m1_status": "bad_token"}
    wt_aa = mut[0]
    try:
        pos = int(mut[1:-1])
    except ValueError:
        return {"m1_status": "bad_token"}
    acc, seq = fetch_uniprot_seq(gene, cache)
    if seq is None:
        return {"m1_uniprot_acc": acc, "m1_status": "no_canonical_seq"}
    if pos < 1 or pos > len(seq):
        return {"m1_uniprot_acc": acc, "m1_canonical_aa": "",
                "m1_status": "position_out_of_range"}
    can = seq[pos - 1]
    return {"m1_uniprot_acc": acc, "m1_canonical_aa": can,
            "m1_status": "match" if can == wt_aa else "MISMATCH"}


# ---------------------------------------------------------------- significance


def mahal_sq(p, mu, cov):
    diff = p - mu
    inv = np.linalg.pinv(cov)
    return float(diff @ inv @ diff)


def permutation_pvalue(wt, mut, n_perm=5000, seed=25):
    rng = np.random.default_rng(seed)
    n_wt, n_mut = len(wt), len(mut)
    pooled = np.vstack([wt, mut])
    obs = float(np.linalg.norm(pooled[:n_wt].mean(0) - pooled[n_wt:].mean(0)))
    if n_wt + n_mut < 4:
        return 1.0, obs
    null = np.zeros(n_perm)
    for i in range(n_perm):
        idx = rng.permutation(n_wt + n_mut)
        null[i] = float(np.linalg.norm(
            pooled[idx[:n_wt]].mean(0) - pooled[idx[n_wt:]].mean(0)))
    return float((null >= obs).sum() + 1) / (n_perm + 1), obs


def bootstrap_delta(wt, mut, n_boot=2000, seed=25):
    rng = np.random.default_rng(seed)
    n_wt, n_mut = len(wt), len(mut)
    if min(n_wt, n_mut) < 2:
        return float("nan"), float("nan"), float("nan")
    boots = np.zeros(n_boot)
    for i in range(n_boot):
        a = wt[rng.integers(0, n_wt, size=n_wt)].mean(0)
        b = mut[rng.integers(0, n_mut, size=n_mut)].mean(0)
        boots[i] = float(np.linalg.norm(a - b))
    return float(np.median(boots)), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mutations-csv", required=True, type=Path)
    ap.add_argument("--latent-csv", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--min-mut-chains", type=int, default=2)
    ap.add_argument("--min-wt-chains", type=int, default=3)
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--skip-m1", action="store_true",
                    help="Skip UniProt M1 (faster, useful for re-runs).")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    muts = pd.read_csv(args.mutations_csv, keep_default_na=False)
    lat = pd.read_csv(args.latent_csv, keep_default_na=False)
    muts["chain_key"] = muts["chain_key"].astype(str).str.upper()
    lat["chain_key"] = lat["chain_key"].astype(str).str.upper()

    # Enumerate mutations per chain.
    print(f"Parsing mutation tokens from {len(muts)} chains")
    chain_to_muts = {}
    for _, r in muts.iterrows():
        chain_to_muts[r["chain_key"]] = extract_mutations_for_chain(r)

    # Build per-(gene, mutation) chain list.
    per_pair: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"chains": set(), "pdbs": set()})
    for _, r in muts.iterrows():
        gene = r["gene"]
        if not gene: continue
        for tok in chain_to_muts[r["chain_key"]]:
            key = (gene, tok)
            per_pair[key]["chains"].add(r["chain_key"])
            per_pair[key]["pdbs"].add(r["pdb"])

    # Filter to pairs with enough mutant chains.
    candidates = []
    for (gene, mut), info in per_pair.items():
        if len(info["chains"]) >= args.min_mut_chains:
            candidates.append({
                "gene": gene, "mutation": mut,
                "n_mut_chains": len(info["chains"]),
                "n_mut_pdbs": len(info["pdbs"]),
                "mut_chain_keys": ";".join(sorted(info["chains"]))})
    print(f"Candidate (gene, mutation) pairs with n>={args.min_mut_chains}: {len(candidates)}")

    df = muts.merge(lat[["chain_key", "z0", "z1"]], on="chain_key", how="left")
    df["z0"] = pd.to_numeric(df["z0"], errors="coerce")
    df["z1"] = pd.to_numeric(df["z1"], errors="coerce")
    df = df[df["z0"].notna() & df["z1"].notna()].copy()

    rows = []
    cache_uniprot = {}
    for i, c in enumerate(candidates):
        if i % 25 == 0:
            print(f"  testing pair {i+1}/{len(candidates)}: {c['gene']} {c['mutation']}")
        gene = c["gene"]; mut = c["mutation"]
        sub = df[df["gene"] == gene]
        wt = sub[~sub["has_any_mutation_annotation"]][["z0", "z1"]].to_numpy(float)
        mut_chain_keys = set(c["mut_chain_keys"].split(";"))
        mut_chains = sub[sub["chain_key"].isin(mut_chain_keys)]
        mu_arr = mut_chains[["z0", "z1"]].to_numpy(float)

        result = {"gene": gene, "mutation": mut,
                  "gene_mut_token": f"{gene}:{mut}",
                  "n_wt_chains": len(wt),
                  "n_mut_chains": c["n_mut_chains"],
                  "n_mut_pdbs": c["n_mut_pdbs"],
                  "mut_chain_keys": c["mut_chain_keys"]}

        if len(wt) < args.min_wt_chains or len(mu_arr) < 1:
            result["comment"] = "too few chains for significance test"
            rows.append(result); continue

        cov = np.cov(wt.T) if len(wt) > 1 else np.eye(2)
        mu_wt = wt.mean(0); mu_mu = mu_arr.mean(0)
        delta = float(np.linalg.norm(mu_wt - mu_mu))
        ms = mahal_sq(mu_mu, mu_wt, cov)
        sigma_wt = float(np.sqrt(np.trace(cov)))
        pval, _ = permutation_pvalue(wt, mu_arr, n_perm=args.n_perm)
        boot_med, boot_lo, boot_hi = bootstrap_delta(
            wt, mu_arr, n_boot=args.n_boot)
        result.update({
            "wt_z0_mean": float(mu_wt[0]), "wt_z1_mean": float(mu_wt[1]),
            "mut_z0_mean": float(mu_mu[0]), "mut_z1_mean": float(mu_mu[1]),
            "delta_latent": delta,
            "sigma_wt_combined": sigma_wt,
            "delta_in_wt_sigmas": delta / sigma_wt if sigma_wt > 0 else float("nan"),
            "mahalanobis_sq": ms,
            "mahalanobis_sigma": float(np.sqrt(ms)),
            "perm_pvalue": pval,
            "boot_delta_median": boot_med,
            "boot_delta_2.5pct": boot_lo,
            "boot_delta_97.5pct": boot_hi,
            "significant_perm_p<0.05": pval < 0.05,
            "significant_mahal_chi2_0.99": ms > chi2.ppf(0.99, df=2),
        })

        if not args.skip_m1:
            m1 = m1_check(gene, mut, cache_uniprot)
            result.update(m1)

        # Placeholder OncoKB columns
        result.update({
            "oncokb_oncogenic": "",
            "oncokb_level": "",
            "oncokb_highest_sensitive_level": "",
            "oncokb_highest_resistance_level": "",
            "oncokb_drug_context": "",
            "oncokb_lookup_status": "PENDING_API_OR_STATIC",
        })

        rows.append(result)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out / "v9_mutation_validation_skeleton.csv", index=False)
    print(f"\nWrote {args.out / 'v9_mutation_validation_skeleton.csv'}  "
          f"({len(out_df)} candidates)")

    # Quick headline summary
    sig = out_df[out_df.get("significant_perm_p<0.05", False) == True]
    print(f"\nSignificant by permutation p<0.05: {len(sig)}/{len(out_df)}")
    if len(sig):
        cols = ["gene", "mutation", "n_wt_chains", "n_mut_chains",
                "delta_in_wt_sigmas", "mahalanobis_sigma", "perm_pvalue"]
        cols = [c for c in cols if c in sig.columns]
        print(sig.sort_values("mahalanobis_sigma", ascending=False)[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
