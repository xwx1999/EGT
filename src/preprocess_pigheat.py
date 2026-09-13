"""Prepare the public PigHeaT pig data for the EGT benchmark.

The released genotype file is a long semicolon-delimited table. This script
keeps the phenotyped backcross animals, converts the long allele calls to
dosages, applies transparent marker QC, and writes the split CSV files used by
``src/egt.py`` and ``src/baselines.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


TRAITS = ["ADG", "BFT23", "RT23"]
MISSING = {"", "NA", "N/A", ".", "0", "-1"}
SPLITS = ("train", "valid", "test")
DATASET = "PigHeaT"
SOURCE_DOI = "10.57745/TLKLRJ"
SOURCE_URL = "https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi%3A10.57745%2FTLKLRJ"


def normalize_id(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def birth_year(animal_id: str) -> int | None:
    match = re.search(r"(20\d{2})", animal_id)
    return int(match.group(1)) if match else None


def split_ids(ids: list[str], seed: int) -> pd.DataFrame:
    """Use 2014 as the future test cohort and split 2013 for train/validation."""
    frame = pd.DataFrame({"ID": ids})
    frame["birth_year"] = frame["ID"].map(birth_year)
    frame["generation"] = "BC"
    if set(frame["birth_year"].dropna().astype(int)) != {2013, 2014}:
        raise ValueError("Expected PigHeaT BC phenotype IDs from birth years 2013 and 2014 only.")

    old = frame[frame["birth_year"] == 2013].copy()
    future = frame[frame["birth_year"] == 2014].copy()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(old))
    n_train = int(round(len(frame) * 0.70))
    n_train = min(max(n_train, 1), len(old) - 1)
    old["split"] = "valid"
    old.iloc[order[:n_train], old.columns.get_loc("split")] = "train"
    future["split"] = "test"
    result = pd.concat([old, future], ignore_index=True)
    result = result.sort_values(["birth_year", "split", "ID"], kind="mergesort").reset_index(drop=True)
    result.insert(2, "forward_order", np.arange(len(result), dtype=int))
    result.insert(3, "split_method", "birth_year_forward")
    result["test_definition"] = np.where(
        result["split"] == "test",
        "BC animals born in 2014; future relative to 2013 training cohort",
        "BC animals born in 2013",
    )
    return result


def read_phenotypes(raw: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    pheno = pd.read_csv(raw / "phenotypes.txt", sep="\t", na_values=["NA"], dtype={"Animal_id": str})
    pheno = pheno.rename(columns={"Animal_id": "ID"})
    pheno["ID"] = pheno["ID"].map(normalize_id)
    ped = pd.read_csv(raw / "Pedigree_animals.txt", sep=";", dtype=str)
    ped = ped.rename(columns={"Animal_id": "ID", "Genetic Type": "genetic_type", "Generation": "generation"})
    ped["ID"] = ped["ID"].map(normalize_id)
    bc = ped[ped["generation"] == "BC"][["ID", "Sire", "Dam", "Gender", "genetic_type", "generation"]]
    pheno = pheno.merge(bc, on="ID", how="inner", validate="one_to_one")
    pheno["birth_year"] = pheno["ID"].map(birth_year)
    pheno = pheno[pheno["birth_year"].isin([2013, 2014])].copy()
    for trait in TRAITS:
        pheno[trait] = pd.to_numeric(pheno[trait], errors="coerce")
    pheno = pheno.dropna(subset=TRAITS, how="all").reset_index(drop=True)
    # The public phenotype table has 1,149 BC IDs. One ID has no usable value
    # for any released trait and is therefore excluded from prediction inputs.
    if len(pheno) != 1148:
        raise ValueError(f"Expected 1148 BC animals with at least one trait, found {len(pheno)}.")
    return pheno, ped


def load_marker_map(raw: Path) -> pd.DataFrame:
    markers = pd.read_csv(raw / "Map_file.txt", sep=";", dtype=str)
    markers = markers.rename(columns={"SNP.Name": "marker_id", "CHR": "chromosome", "Position": "position"})
    markers["position"] = pd.to_numeric(markers["position"], errors="coerce")
    markers = markers[markers["chromosome"].astype(str).str.fullmatch(r"[1-9]|1[0-8]")].copy()
    markers = markers.drop_duplicates("marker_id").reset_index(drop=True)
    if markers.empty:
        raise ValueError("No autosomal markers were found in Map_file.txt.")
    return markers


def read_long_genotypes(raw: Path, sample_ids: list[str], markers: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Stream the 2.6 GB long table into a compact sample-by-marker matrix."""
    sample_index = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}
    marker_index = {marker_id: idx for idx, marker_id in enumerate(markers["marker_id"].astype(str))}
    matrix = np.full((len(sample_ids), len(markers)), np.nan, dtype=np.float32)
    dosage_allele: list[str | None] = [None] * len(markers)
    seen_rows = 0
    seen_samples: set[str] = set()

    with (raw / "Genotypes.txt").open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.reader(handle, delimiter=";")
        header = next(reader)
        if header[:2] != ["ID", "SNP Name"]:
            raise ValueError(f"Unexpected genotype header: {header[:4]}")
        for row in reader:
            if len(row) < 4:
                continue
            sample_id, marker_id, allele1, allele2 = (token.strip() for token in row[:4])
            sample_row = sample_index.get(sample_id)
            marker_col = marker_index.get(marker_id)
            if sample_row is None or marker_col is None:
                continue
            seen_rows += 1
            seen_samples.add(sample_id)
            if dosage_allele[marker_col] is None:
                dosage_allele[marker_col] = next((a for a in (allele1, allele2) if a not in MISSING), None)
            alt = dosage_allele[marker_col]
            if alt is None:
                continue
            alleles = [allele1, allele2]
            if any(allele in MISSING for allele in alleles):
                continue
            matrix[sample_row, marker_col] = float(sum(allele == alt for allele in alleles))

    if seen_samples != set(sample_ids):
        missing = sorted(set(sample_ids) - seen_samples)
        raise ValueError(f"Genotype rows are missing for {len(missing)} phenotype IDs; first IDs: {missing[:5]}")
    if seen_rows == 0:
        raise ValueError("No phenotype/genotype overlap was found.")
    return matrix, np.asarray(dosage_allele, dtype=object)


def apply_marker_qc(matrix: np.ndarray, markers: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    call_rate = np.isfinite(matrix).mean(axis=0)
    allele_frequency = np.nanmean(matrix, axis=0) / 2.0
    maf = np.minimum(allele_frequency, 1.0 - allele_frequency)
    keep = (call_rate >= 0.95) & np.isfinite(maf) & (maf >= 0.01)
    if int(keep.sum()) < 1000:
        raise ValueError(f"Marker QC retained only {int(keep.sum())} markers.")
    qc = markers.loc[keep].copy().reset_index(drop=True)
    qc["call_rate"] = call_rate[keep]
    qc["maf"] = maf[keep]
    return matrix[:, keep], qc


def write_matrix(path: Path, ids: list[str], matrix: np.ndarray, marker_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ID", *marker_ids])
        for sample_id, values in zip(ids, matrix):
            row = [sample_id]
            row.extend("" if not np.isfinite(value) else str(int(value)) for value in values)
            writer.writerow(row)


def write_json(payload: dict, path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def process_pigheat(raw: Path, out: Path, seed: int = 20260911) -> dict:
    """Build the processed PigHeaT bundle and return its audit summary."""
    out.mkdir(parents=True, exist_ok=True)

    pheno, pedigree = read_phenotypes(raw)
    splits = split_ids(pheno["ID"].tolist(), seed)
    pheno = pheno.merge(splits, on=["ID", "birth_year", "generation"], how="left", validate="one_to_one")
    if pheno["split"].isna().any():
        raise ValueError("Some phenotype IDs did not receive a split.")

    markers = load_marker_map(raw)
    matrix, dosage_alleles = read_long_genotypes(raw, pheno["ID"].tolist(), markers)
    matrix, qc_markers = apply_marker_qc(matrix, markers)
    marker_ids = qc_markers["marker_id"].astype(str).tolist()

    pheno_columns = [
        "ID", "birth_year", "generation", "genetic_type", "Gender", "Sire", "Dam",
        *TRAITS, "split", "forward_order", "split_method", "test_definition",
    ]
    pheno[pheno_columns].to_csv(out / "phenotype_wide.csv", index=False)
    labels = pheno.melt(
        id_vars=["ID", "birth_year", "generation", "split", "forward_order", "split_method"],
        value_vars=TRAITS,
        var_name="trait",
        value_name="value",
    ).dropna(subset=["value"])
    labels.to_csv(out / "labels_long.csv", index=False)
    qc_markers.to_csv(out / "markers.csv", index=False)

    pedigree = pedigree.rename(columns={"Animal_id": "ID", "Genetic Type": "genetic_type", "Generation": "generation"})
    pedigree["ID"] = pedigree["ID"].map(normalize_id)
    pedigree = pedigree.merge(splits[["ID", "split", "forward_order", "split_method"]], on="ID", how="left")
    pedigree.to_csv(out / "pedigree.csv", index=False)

    samples = pheno[["ID", "birth_year", "generation", "genetic_type", "split", "forward_order", "split_method"]].copy()
    samples.to_csv(out / "genotype_samples.csv", index=False)
    splits.to_csv(out / "splits.csv", index=False)

    row_by_id = {sample_id: idx for idx, sample_id in enumerate(pheno["ID"])}
    for split in SPLITS:
        selected = splits[splits["split"] == split]["ID"].tolist()
        indices = np.asarray([row_by_id[sample_id] for sample_id in selected], dtype=int)
        write_matrix(out / f"genotypes_{split}.csv", selected, matrix[indices], marker_ids)

    dosage = pd.DataFrame({"marker_id": markers["marker_id"], "dosage_allele": dosage_alleles})
    dosage = dosage[dosage["marker_id"].isin(set(marker_ids))]
    dosage.to_csv(out / "dosage_alleles.csv", index=False)

    split_counts = splits["split"].value_counts().reindex(SPLITS, fill_value=0).astype(int).to_dict()
    trait_counts = labels.groupby(["trait", "split"]).size().unstack(fill_value=0).reindex(columns=SPLITS, fill_value=0)
    summary = {
        "dataset": DATASET,
        "source_doi": SOURCE_DOI,
        "source_url": SOURCE_URL,
        "source_license": "Etalab Open Licence 2.0 (compatible with CC-BY 2.0)",
        "traits": TRAITS,
        "split_method": "birth_year_forward",
        "split_rule": "All 2014-born BC animals are test; 2013-born BC animals are split into train/valid with seed 20260911.",
        "split_ratio_requested": [7.0, 1.0, 2.0],
        "split_counts": split_counts,
        "split_percentages": {key: round(value / len(splits) * 100.0, 3) for key, value in split_counts.items()},
        "birth_year_by_split": splits.groupby("split")["birth_year"].agg(["min", "max", "nunique"]).reindex(SPLITS).fillna(0).to_dict("index"),
        "n_genotype_samples": int(len(pheno)),
        "n_raw_bc_phenotype_ids": 1149,
        "n_bc_ids_without_any_trait": 1,
        "n_markers_before_qc": int(len(markers)),
        "n_markers_after_qc": int(len(qc_markers)),
        "marker_qc": {"chromosomes": "1-18", "call_rate_min": 0.95, "maf_min": 0.01},
        "trait_counts": {trait: {split: int(trait_counts.loc[trait, split]) for split in SPLITS} for trait in TRAITS},
        "genotype_encoding": "0/1/2 dosage of the first observed allele per marker; marker orientation does not affect prediction.",
        "seed": seed,
    }
    write_json(summary, out / "summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess public PigHeaT data for strict birth-year forward validation.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/PigHeaT"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/PigHeaT"))
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()
    summary = process_pigheat(args.raw_dir, args.out_dir, args.seed)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
