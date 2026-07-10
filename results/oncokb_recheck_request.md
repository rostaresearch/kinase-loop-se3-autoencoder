# OncoKB re-check — SE(3) mutation-significance hits + featured drivers

_Auto-generated from `v91_significance_summary.csv` (SE(3), 16 perm-p<0.05 hits) merged with `v91_mutation_validation_with_oncokb_api.csv`. Please tick / correct the last two columns._

**What we need:** for each row, confirm (a) the OncoKB call is current, and (b) whether the underlying PDB structures are a **genuine disease mutant** or an **engineered construct / phospho-mimetic** (which we must NOT quote as a disease result).

Legend for our guessed category: `disease?` = candidate real allele · `phospho` = engineered activation-loop phospho-mimetic/-dead (expected to remodel the loop, not a disease claim) · `construct` = crystallization / kinase-dead engineering.


## A. OncoKB-annotated (verify annotation + PDB provenance)

| gene | mutation | σ (Mahal) | perm p | #PDBs | OncoKB oncogenic | OncoKB effect | level (sens/res) | our guess | **CONFIRM: real disease? (Y/N)** | notes |
|---|---|--:|--:|--:|---|---|---|---|---|---|
| FGFR3 | V555M | 7.76 | 0.0391 | 1 | Likely Oncogenic | Likely Gain-of-function | LEVEL_2/- | disease? |  |  |
| EGFR | T790M | 2.37 | 9.999e-05 | 53 | Oncogenic | Gain-of-function | LEVEL_1/LEVEL_R1 | disease? |  |  |
| BRAF | V600E | 1.42 | 0.0381 | 2 | Oncogenic | Gain-of-function | LEVEL_1/- | disease? |  |  |
| MET | D1228V | 0.98 | 0.032 | 5 | Inconclusive | Inconclusive | -/- | disease? |  |  |
| FGFR2 | K659N | 0.84 | 0.0398 | 2 | Likely Oncogenic | Gain-of-function | LEVEL_2/- | disease? |  |  |

## B. Engineered phospho-mimetic / -dead — activation loop (confirm construct, exclude from disease claims)

| gene | mutation | σ (Mahal) | perm p | #PDBs | OncoKB oncogenic | OncoKB effect | level (sens/res) | our guess | **CONFIRM: real disease? (Y/N)** | notes |
|---|---|--:|--:|--:|---|---|---|---|---|---|
| MARK2 | T208E | 290.69 | 0.005799 | 1 | Unknown | Unknown | -/- | phospho |  |  |
| DAPK2 | S308D | 3.19 | 9.999e-05 | 1 | Unknown | Unknown | -/- | phospho |  |  |
| GRK1 | T489E | 2.49 | 0.0119 | 3 | Unknown | Unknown | -/- | phospho |  |  |
| GRK1 | S488E | 2.49 | 0.0119 | 3 | Unknown | Unknown | -/- | phospho |  |  |
| DAPK2 | S308A | 1.35 | 0.0002 | 2 | Unknown | Unknown | -/- | phospho |  |  |

## C. Suspected crystallization / kinase-dead constructs (confirm & exclude)

| gene | mutation | σ (Mahal) | perm p | #PDBs | OncoKB oncogenic | OncoKB effect | level (sens/res) | our guess | **CONFIRM: real disease? (Y/N)** | notes |
|---|---|--:|--:|--:|---|---|---|---|---|---|
| MARK2 | K82R | 290.69 | 0.005799 | 1 | Unknown | Unknown | -/- | construct |  |  |
| EGFR | V948R | 3.92 | 9.999e-05 | 28 | Unknown | Unknown | -/- | construct |  |  |

## D. Uncatalogued / unclear — please classify

| gene | mutation | σ (Mahal) | perm p | #PDBs | OncoKB oncogenic | OncoKB effect | level (sens/res) | our guess | **CONFIRM: real disease? (Y/N)** | notes |
|---|---|--:|--:|--:|---|---|---|---|---|---|
| PRP4K | L715F | 26.1 | 0.0024 | 1 | Unknown | Unknown | -/- | disease? |  |  |
| GRK5 | D311N | 2.84 | 0.0041 | 6 | Unknown | Unknown | -/- | disease? |  |  |
| DAPK2 | W305S | 1.35 | 0.0002 | 2 | Unknown | Unknown | -/- | disease? |  |  |
| PIM1 | R250G | 0.94 | 0.032 | 44 | Unknown | Unknown | -/- | disease? |  |  |

## Statistical-artifact flags (FYI, not OncoKB)

- **MARK2 T208E / K82R (σ≈291)** and **PRP4K L715F (σ≈26)** come from a *single* PDB each (2WZJ, 7Q4A). The huge σ is a degenerate-covariance artifact (mutant chains near-identical → tiny within-group variance), not a large real shift. We will down-weight these regardless of OncoKB.


## PDB IDs per mutation (for provenance checks)

- **EGFR V948R** (construct): 4I1ZA,4I20A,5HG5A,5HG8A,5HG9A,5UG8A,5UG9A,5UGCA,5X2AB,5X2FB,5X2FD,6P8QC,6V5NA,6V5ND,6V5PA,6V5PD,6V66A,6V66D,6V6OA,6V6OB,6V6OD,6WA2B,6WA2D,6WAKB,6XL4C,7A6IA,7JXPA,7JXPB,7LGSA,7LGSB,7LGSC,7LGSD,7T4JA,7T4JB,7UKWA,7UKWD,8D73A,8D73B,8D76A,8D76B,8F1WA,8F1WD,8FV4B,8PO4A
- **MARK2 K82R** (construct): 2WZJA,2WZJB,2WZJC,2WZJD,2WZJE,2WZJF
- **BRAF V600E** (disease?): 7P3VB,8C7YA
- **DAPK2 W305S** (disease?): 1Z9XA,1Z9XB,1Z9XC,2A27A,2A27B,2A27C,2A27D,2A27E,2A27F,2A27G,2A27H
- **EGFR T790M** (disease?): 2JIUA,2JIUB,3IKAA,3IKAB,3UG1A,3VJNA,3W2OA,3W2PA,4G5PA,4G5PB,4I1ZA,4I21A,4I21B,4WD5A,4WD5B,5FEEA,5FEQA,5GTZA,5HG5A,5HG8A,5HG9A,5HIBA,5J9YA,5UG8A,5UG9A,5UGCA,5X2AB,5X2FB,5X2FD,5X2KA,5XDKA,5Y9TA,6JX0A,6JX4A,6LUBA,6LUDA,6P8QC,6S89A,6S8AA,6V5NA,6V5ND,6V5PA,6V5PD,6V66A,6V66D,6V6OA,6V6OB,6V6OD,6WA2B,6WA2D,6WAKB,6XL4C,7A6IA,7JXPA,7JXPB,7OXBA,7UKWA,7UKWD,7VREA,7ZYMA,7ZYNA,7ZYPA,8D73A,8D73B,8D76A,8D76B,8F1WA,8F1WD,8FV4B,8H7XA,8H7XB
- **FGFR2 K659N** (disease?): 2PVYA,2PVYB,2PVYC,2PVYD,4J95A,4J95B,4J95C
- **FGFR3 V555M** (disease?): 8UDVA,8UDVB
- **GRK5 D311N** (disease?): 9BREA,9CKPA,9CKQA,9CKRA,9CKSA,9MX2A
- **MET D1228V** (disease?): 6SDCA,6SDDA,8OUUA,8OUUB,8OUVA,8OUVB,8OVZA,8OVZB
- **PIM1 R250G** (disease?): 2BZHB,2BZIB,2BZJA,2BZKB,2C3IB,2J2IB,3BWFA,3CXWA,3CY2A,3CY3A,3JPVA,3MA3A,3QF9A,3WE8A,4AS0A,4GW8A,5MZLA,5N4NA,5N4OA,5N4RA,5N4UA,5N4VA,5N4XA,5N4ZA,5N51A,5N52A,5N5LA,5N5MA,6AYDA,7Z6UA,8AFRA,8R0HA,8R0QA,8R0WA,8R0YA,8R10A,8R18A,8R1KA,8R1NA,8R1PA,8R1TA,8R1WA,8R25A,8R27A
- **PRP4K L715F** (disease?): 7Q4AA,7Q4AB
- **DAPK2 S308D** (phospho): 1WMKA,1WMKB,1WMKC,1WMKD,1WMKE,1WMKF,1WMKG,1WMKH
- **DAPK2 S308A** (phospho): 1Z9XA,1Z9XB,1Z9XC,2A27A,2A27B,2A27C,2A27D,2A27E,2A27F,2A27G,2A27H
- **GRK1 T489E** (phospho): 7MT8G,7MTAG,7MTBG
- **GRK1 S488E** (phospho): 7MT8G,7MTAG,7MTBG
- **MARK2 T208E** (phospho): 2WZJA,2WZJB,2WZJC,2WZJD,2WZJE,2WZJF