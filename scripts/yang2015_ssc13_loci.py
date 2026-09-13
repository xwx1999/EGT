"""Extract the SSC13 loci reported by Yang et al. (2015) from their supplementary tables.

Animal QTLdb's live overlap query could not be run (animalgenome.org, faang.org
and the genome.iastate.edu mirror were all serving a maintenance page, and
web.archive.org is unreachable from this environment). The reviewer's request is
for a benchmark against the published GWAS on these same animals, and the study's
own supplementary tables provide the genome-based, base-pair positions directly:

* S4 Table - SNPs above the suggestive significance level in the three populations
* S5 Table - SNPs above the suggestive level in the five-population meta-analysis

Both are retrieved from the EBI BioStudies mirror of the PLOS article, which is
reachable where the journal's own object store is not.

Run from the project root:
    python scripts/yang2015_ssc13_loci.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUPP = ROOT / "tmp" / "yang2015_supp"
OUT = ROOT / "results" / "animals_round1" / "ssc13_annotation"
MARKER = "MARC0013088"
MARKER_POS = 140_497_749
TARGET_CHROM = 13


def read_table(path: Path, skip: int) -> pd.DataFrame:
    df = pd.read_excel(path, skiprows=skip)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    s4 = read_table(SUPP / "pone.0131667.s007.xls", skip=1)
    s5 = read_table(SUPP / "pone.0131667.s008.xls", skip=1)
    print("=" * 84)
    print("Yang et al. (2015) GWAS loci - supplementary tables")
    print("=" * 84)
    print(f"S4 (three populations) rows: {len(s4)}   columns: {list(s4.columns)}")
    print(f"S5 (meta-analysis)     rows: {len(s5)}   columns: {list(s5.columns)}")
    print()

    s4["Chromosome"] = pd.to_numeric(s4["Chromosome"], errors="coerce")
    s5["Chromosome"] = pd.to_numeric(s5["Chromosome"], errors="coerce")

    for name, df in (("S4 (three populations)", s4), ("S5 (meta-analysis)", s5)):
        sub = df[df["Chromosome"] == TARGET_CHROM].copy()
        print(f"--- {name}: SSC13 rows = {len(sub)} ---")
        if len(sub):
            pos = pd.to_numeric(sub["Position"], errors="coerce")
            print(f"    SSC13 position range: {pos.min():,.0f} - {pos.max():,.0f}")
            cols = [c for c in ["Trait", "Breed", "Illumina SNP", "Significant SNP", "Position",
                                "P-value", "q-value", "Significance threshold", "Candidate gene"] if c in sub.columns]
            print(sub[cols].to_string(index=False))
        print()

    # all loci across the genome, for the APOB/SSC3 comparison
    print("--- SSC3 rows (the study's headline APOB region) ---")
    for name, df in (("S4", s4), ("S5", s5)):
        sub = df[df["Chromosome"] == 3].copy()
        if len(sub):
            pos = pd.to_numeric(sub["Position"], errors="coerce")
            print(f"    {name}: n={len(sub)}  range {pos.min():,.0f}-{pos.max():,.0f}  traits={sorted(sub['Trait'].unique())}")
    print()

    # does the attributed marker itself appear?
    for name, df in (("S4", s4), ("S5", s5)):
        cols = [c for c in ("Illumina SNP", "Significant SNP") if c in df.columns]
        hits = df[df[cols].apply(lambda r: r.astype(str).str.contains(MARKER, case=False, na=False).any(), axis=1)]
        print(f"{MARKER} reported in {name}: {len(hits)}")
    print()

    # nearest reported locus to the attributed marker, genome-wide
    print(f"--- nearest reported SNP to {MARKER} (SSC13:{MARKER_POS:,}) ---")
    nearest = []
    for name, df in (("S4", s4), ("S5", s5)):
        sub = df[df["Chromosome"] == TARGET_CHROM].copy()
        if not len(sub):
            continue
        sub["Position"] = pd.to_numeric(sub["Position"], errors="coerce")
        sub["distance_bp"] = (sub["Position"] - MARKER_POS).abs()
        row = sub.nsmallest(1, "distance_bp").iloc[0]
        nearest.append({
            "table": name, "snp": str(row.get("Illumina SNP", row.get("Significant SNP"))),
            "trait": row["Trait"], "position": int(row["Position"]),
            "distance_bp": int(row["distance_bp"]),
        })
        print(f"  {name}: {row.get('Illumina SNP', row.get('Significant SNP'))} at {int(row['Position']):,} "
              f"({row['Trait']}), {int(row['distance_bp']):,} bp away")
    print()

    ssc13_s4 = s4[s4["Chromosome"] == TARGET_CHROM]
    ssc13_s5 = s5[s5["Chromosome"] == TARGET_CHROM]
    if len(ssc13_s4):
        ssc13_s4.to_csv(OUT / "yang2015_ssc13_snps_S4.csv", index=False)
    if len(ssc13_s5):
        ssc13_s5.to_csv(OUT / "yang2015_ssc13_snps_S5.csv", index=False)
    s4.to_csv(OUT / "yang2015_all_snps_S4.csv", index=False)
    s5.to_csv(OUT / "yang2015_all_snps_S5.csv", index=False)

    summary = {
        "source": "Yang et al. (2015) PLoS ONE 10(6):e0131667, supplementary S4 and S5 tables",
        "retrieved_from": "EBI BioStudies mirror (S-EPMC4488070)",
        "note_on_qtldb": (
            "Animal QTLdb live overlap query could not be run; animalgenome.org, faang.org "
            "and genome.iastate.edu were all under maintenance and web.archive.org is "
            "unreachable from this environment. These tables are the primary publication "
            "underlying the BloodLipid dataset and carry genome-based bp positions."
        ),
        "s4_rows": int(len(s4)),
        "s5_rows": int(len(s5)),
        "ssc13_rows_s4": int(len(ssc13_s4)),
        "ssc13_rows_s5": int(len(ssc13_s5)),
        "ssc13_snps_s4": ssc13_s4[[c for c in ("Trait", "Breed", "Illumina SNP", "Position", "P-value", "Candidate gene") if c in ssc13_s4.columns]].to_dict(orient="records"),
        "candidate_marker": MARKER,
        "candidate_marker_position": MARKER_POS,
        "nearest_reported_to_marker": nearest,
    }
    (OUT / "yang2015_ssc13_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"written: {OUT / 'yang2015_all_snps_S4.csv'}")
    print(f"written: {OUT / 'yang2015_ssc13_summary.json'}")


if __name__ == "__main__":
    main()
