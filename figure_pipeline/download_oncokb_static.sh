#!/bin/bash
# =====================================================================
# Download the OncoKB static TSVs required by
#   join_oncokb_to_skeleton.py  --oncokb-static-dir <OUT>
#
# It needs three files (the script globs for these name patterns):
#   allActionableVariants.tsv   <- OncoKB "Actionable Genes"   (needs a token)
#   allCuratedVariants.tsv      <- OncoKB "Annotated Variants" (needs a token)
#   allCuratedGenes.tsv         <- OncoKB "Cancer Gene List"   (PUBLIC, no token)
#
# TOKEN: the two variant files need a personal OncoKB API token. It is free for
# academics — register at  https://www.oncokb.org/account/register  then copy your
# token from  https://www.oncokb.org/account/settings.  Save it to a file
# (chmod 600) and pass it below, or export ONCOKB_TOKEN.
#
# Usage:
#   ./download_oncokb_static.sh <out-dir> [token-file]
#   # examples:
#   ./download_oncokb_static.sh ./oncokb ~/.oncokb_token
#   ONCOKB_TOKEN=xxxxxxxx ./download_oncokb_static.sh ./oncokb
# =====================================================================
set -euo pipefail

OUT="${1:?usage: download_oncokb_static.sh <out-dir> [token-file]}"
TOKENFILE="${2:-$HOME/.oncokb_token}"
BASE="https://www.oncokb.org/api/v1/utils"
mkdir -p "$OUT"

# resolve token (env var wins; else read the file). Only needed for the 2 variant files.
TOKEN="${ONCOKB_TOKEN:-}"
if [ -z "$TOKEN" ] && [ -f "$TOKENFILE" ]; then
  TOKEN="$(tr -d '[:space:]' < "$TOKENFILE")"
fi

get() {  # get <endpoint> <outfile> <needs_token 0|1>
  local ep="$1" out="$2" need="$3" auth=()
  if [ "$need" = 1 ]; then
    if [ -z "$TOKEN" ]; then
      echo "!! $out needs a token but none found (set ONCOKB_TOKEN or pass a token file). Skipping." >&2
      return 1
    fi
    auth=(-H "Authorization: Bearer $TOKEN")
  fi
  echo "downloading $out ..."
  local code
  code=$(curl -sS "${auth[@]}" -o "$OUT/$out" -w '%{http_code}' "$BASE/$ep")
  if [ "$code" != 200 ]; then
    echo "!! $out: HTTP $code (401 = bad/expired token; 429 = rate-limited)" >&2
    return 1
  fi
  echo "   OK  $(wc -l < "$OUT/$out") lines -> $OUT/$out"
}

rc=0
get allActionableVariants.txt  allActionableVariants.tsv 1 || rc=1
get allAnnotatedVariants.txt   allCuratedVariants.tsv    1 || rc=1
get cancerGeneList.txt         allCuratedGenes.tsv       0 || rc=1

echo
echo "Files now in $OUT :"
ls -la "$OUT"/*.tsv 2>/dev/null || true
if [ "$rc" = 0 ]; then
  echo "All three downloaded. Point the pipeline at it:  --oncokb-static-dir $OUT"
else
  echo "Some files were skipped (see messages above) — most likely the token." >&2
fi
exit $rc
