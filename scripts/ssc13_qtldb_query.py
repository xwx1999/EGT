"""Extract SSC13 and blood-lipid QTL records from the Animal QTLdb legacy deposition.

Animal QTLdb was unreachable during this revision (animalgenome.org, faang.org and
the genome.iastate.edu mirror all served a maintenance page, and web.archive.org
is not reachable from this environment). The database does publish its retired
linkage-map-based QTL as a permanent deposition in the AnimalGenome/QTLdb GitHub
repository, which is reachable, so the QTLdb query is answered from that source.

Scope note that must travel with any citation of this file: it contains the
**linkage-map-based** QTL curated 1999-2021, positioned in cM on the Rohrer et al.
(1996) porcine linkage map. QTLdb's genome-based QTL and SNP-association records
were not part of this deposition and could not be retrieved while the site is
down. Positions are therefore marker-anchored rather than base-pair coordinates,
and cannot be intersected directly with Sscrofa10.2 sequence coordinates.

Run from the project root:
    python scripts/ssc13_qtldb_query.py
"""
from __future__ import annotations

import csv
import gzip
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "tmp" / "qtldb" / "QTLdb_legacyQTLdata.pig.txt.gz"
OUT = ROOT / "results" / "animals_round1" / "ssc13_annotation"
TARGET_CHROM = "13"
# "serum"/"plasma" are deliberately absent: they also match immunological traits
# such as "Mycoplasma hyopneumoniae antibody titer", which are not lipid records.
LIPID_TERMS = [
    "lipid", "cholesterol", "triglyceride", "fatty acid", "lipoprotein",
    "hdl", "ldl", "vldl", "adipos", "fat",
]


def load() -> pd.DataFrame:
    """Read the deposition; the header carries one stray trailing empty field.

    Header line has 30 tab-separated entries but every data row has 29, the last
    header entry being empty. Columns are therefore taken from the header up to
    the real width of the data.
    """
    with gzip.open(RAW, "rt", encoding="utf-8", errors="replace") as fh:
        rows = [line.rstrip("\n") for line in fh if not line.startswith("##")]
    header = [h for h in rows[0].lstrip("#").split("\t")]
    body = [r.split("\t") for r in rows[1:]]
    width = len(body[0])
    if any(len(r) != width for r in body):
        raise SystemExit("Ragged rows in the QTLdb deposition; refusing to guess alignment.")
    if len(header) == width + 1 and header[-1] == "":
        header = header[:-1]
    if len(header) != width:
        raise SystemExit(f"Header width {len(header)} != data width {width}.")
    return pd.DataFrame(body, columns=header)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    print(f"QTLdb legacy pig records: {len(df)}")
    print(f"columns: {list(df.columns)}")
    print()

    df["Chromosome"] = df["Chromosome"].astype(str).str.strip().str.lower()
    ssc13 = df[df["Chromosome"] == TARGET_CHROM].copy()
    print(f"SSC13 records: {len(ssc13)}")

    trait_col = "Reported trait"
    ssc13["_trait_lower"] = ssc13[trait_col].astype(str).str.lower()
    lipid_mask = ssc13["_trait_lower"].apply(lambda t: any(term in t for term in LIPID_TERMS))
    ssc13_lipid = ssc13[lipid_mask].copy()

    print(f"SSC13 records with a lipid/fat-related trait: {len(ssc13_lipid)}")
    print()
    print("--- distinct SSC13 lipid-related traits ---")
    counts = ssc13_lipid[trait_col].value_counts()
    for trait, n in counts.items():
        print(f"  {n:>3}  {trait}")

    cols = ["QTL_ID", trait_col, "Breeds", "Map location", "Left marker (suggestive level)",
            "Right marker (suggestive level)", "Pubmed ID"]
    cols = [c for c in cols if c in ssc13_lipid.columns]
    keep = ssc13_lipid[cols].copy()
    keep.to_csv(OUT / "ssc13_qtldb_qtl_records.csv", index=False)
    ssc13[cols].to_csv(OUT / "ssc13_qtldb_all_records.csv", index=False)

    print()
    print("--- SSC13 lipid-related QTL records (map location in cM) ---")
    print(keep.to_string(index=False, max_colwidth=44))

    summary = {
        "source": "Animal QTLdb legacy linkage-map-based QTL deposition",
        "source_url": "https://github.com/AnimalGenome/QTLdb",
        "retrieved": "2026-09-12",
        "scope_caveat": (
            "Linkage-map-based QTL only, curated 1999-2021, positioned in cM on the "
            "Rohrer et al. (1996) porcine linkage map. Genome-based QTL and "
            "SNP-association records were not retrievable because animalgenome.org, "
            "faang.org and the genome.iastate.edu mirror were all under maintenance."
        ),
        "records_total": int(len(df)),
        "records_ssc13": int(len(ssc13)),
        "records_ssc13_lipid_related": int(len(ssc13_lipid)),
        "ssc13_lipid_traits": {str(k): int(v) for k, v in counts.items()},
        "candidate_marker": "MARC0013088",
        "candidate_marker_coordinate": "SSC13:140,497,749 (Sscrofa10.2)",
    }
    import json
    (OUT / "ssc13_qtldb_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print()
    print(f"written: {OUT / 'ssc13_qtldb_qtl_records.csv'}")
    print(f"written: {OUT / 'ssc13_qtldb_summary.json'}")


if __name__ == "__main__":
    main()
