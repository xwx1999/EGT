from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


MISSING_VALUES = ["", ".", "NA", "N/A", "NaN", "nan", "NULL", "null"]
SPLIT_LABELS = ("train", "valid", "test")


def normalize_id(value: object) -> str:
    """Normalize animal/sample identifiers into a consistent string form.

    The raw files mix integer IDs, floating-point-looking IDs from spreadsheets
    (for example ``3544.0``), and plain text IDs. This function removes
    whitespace, converts whole-number floats to integer strings, and returns an
    empty string for missing values so downstream joins use one stable ID format.
    """
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        number = float(text)
    except ValueError:
        return text[:-2] if text.endswith(".0") else text
    if math.isfinite(number) and number.is_integer():
        return str(int(number))
    return text


def normalize_id_column(df: pd.DataFrame, column: str = "ID") -> pd.DataFrame:
    """Return a copy of ``df`` with one ID column normalized and empty IDs removed.

    All joins between phenotype, genotype, pedigree, and split tables depend on
    exact ID matches. Normalizing the column once at ingestion prevents subtle
    mismatches such as ``153`` versus ``153.0``.
    """
    df = df.copy()
    df[column] = df[column].map(normalize_id)
    return df[df[column] != ""].copy()


def numeric_id_sort(series: pd.Series) -> pd.Series:
    """Convert an ID series to numeric values for chronological-like sorting.

    Some datasets do not provide usable time or pedigree information. In those
    cases the requested fallback is an ID-based forward split, so this helper
    gives numeric IDs their natural order while leaving non-numeric IDs as NaN.
    """
    return pd.to_numeric(series, errors="coerce")


def coerce_numeric(df: pd.DataFrame, exclude: Iterable[str] = ("ID",)) -> pd.DataFrame:
    """Convert all non-excluded columns to numeric values where possible.

    Raw phenotype and EBV files store missing values as strings such as ``.``.
    This helper standardizes trait columns to numeric dtypes and turns invalid
    entries into NaN, while preserving ID-like columns as strings.
    """
    df = df.copy()
    excluded = set(exclude)
    for column in df.columns:
        if column not in excluded:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to CSV, creating the output directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_json(payload: dict, path: Path) -> None:
    """Write a JSON metadata file with readable indentation and UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_wide_genotype_ids(path: Path, delimiter: str) -> list[str]:
    """Read only sample IDs from a wide genotype matrix without loading markers.

    HZA and PIC genotype files are very wide, with tens of thousands of marker
    columns and in PIC's case close to a gigabyte of raw text. For split
    creation we only need the first column, so this function streams line by
    line and avoids loading the full genotype matrix into memory.
    """
    ids: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        next(handle)
        for line in handle:
            if not line.strip():
                continue
            if delimiter == ",":
                token = line.split(",", 1)[0]
            else:
                token = line.split(None, 1)[0]
            ids.append(normalize_id(token))
    return ids


def read_wide_genotype_header(path: Path, delimiter: str) -> list[str]:
    """Read marker names from the header row of a wide genotype matrix.

    The first column is normalized to ``ID`` and the remaining columns are
    treated as marker IDs. This lets the script export marker metadata without
    parsing every genotype value.
    """
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        line = handle.readline().strip()
    tokens = line.split(",") if delimiter == "," else line.split()
    tokens[0] = "ID"
    return tokens


def read_plink_ped_prefix(path: Path) -> pd.DataFrame:
    """Read the six metadata columns from a PLINK PED file.

    A PLINK PED row starts with family ID, individual ID, sire ID, dam ID, sex,
    and phenotype, followed by two allele columns per marker. This function
    extracts only the prefix needed for sample metadata and pedigree-aware
    processing, which avoids loading a very wide PED genotype matrix.
    """
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            tokens = line.split()
            if len(tokens) >= 6:
                rows.append(tokens[:6])
    df = pd.DataFrame(rows, columns=["FID", "ID", "SIRE", "DAM", "SEX", "PED_PHENOTYPE"])
    for column in df.columns:
        df[column] = df[column].map(normalize_id)
    return df


def split_counts(n_items: int, ratio: tuple[float, float, float]) -> tuple[int, int, int]:
    """Compute train/validation/test counts from a ratio and item count.

    The first boundary is floor(``n * train_ratio``), the second boundary is
    floor(``n * (train_ratio + valid_ratio)``), and the remainder goes to test.
    This keeps all samples assigned exactly once even when counts do not divide
    cleanly by the ratio.
    """
    total = sum(ratio)
    train_n = int(math.floor(n_items * ratio[0] / total))
    valid_n = int(math.floor(n_items * (ratio[0] + ratio[1]) / total)) - train_n
    test_n = n_items - train_n - valid_n
    return train_n, valid_n, test_n


def make_forward_split(
    entities: pd.DataFrame,
    id_col: str,
    split_method: str,
    sort_cols: list[str],
    ratio: tuple[float, float, float],
) -> pd.DataFrame:
    """Create a deterministic forward train/valid/test split.

    ``entities`` should contain one row per splitting unit, usually one animal
    ID. The rows are sorted by the supplied forward-order columns, then by
    numeric/text ID as deterministic tie-breakers. Earlier rows become train,
    the next block becomes validation, and the latest rows become test.
    """
    entities = entities.drop_duplicates(id_col).copy()
    entities["_id_num_sort"] = numeric_id_sort(entities[id_col])
    entities["_id_text_sort"] = entities[id_col].astype(str)
    sort_by = sort_cols + ["_id_num_sort", "_id_text_sort"]
    entities = entities.sort_values(sort_by, kind="mergesort", na_position="last").reset_index(drop=True)

    train_n, valid_n, _ = split_counts(len(entities), ratio)
    split = np.repeat("test", len(entities)).astype(object)
    split[:train_n] = "train"
    split[train_n : train_n + valid_n] = "valid"

    entities.insert(1, "split", split)
    entities.insert(2, "forward_order", np.arange(len(entities), dtype=int))
    entities.insert(3, "split_method", split_method)
    entities = entities.drop(columns=["_id_num_sort", "_id_text_sort"])
    return entities


def split_summary(split_df: pd.DataFrame) -> dict:
    """Return split counts as a stable train/valid/test dictionary."""
    counts = split_df["split"].value_counts().reindex(SPLIT_LABELS, fill_value=0)
    return {label: int(counts[label]) for label in SPLIT_LABELS}


def select_forward_split(
    entities: pd.DataFrame,
    id_col: str,
    ratio: tuple[float, float, float],
    time_cols: list[str] | None = None,
    pedigree_col: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """Choose the best available forward split strategy.

    The requested priority is: use time when usable, otherwise use pedigree
    generation, otherwise fall back to ID order. The function returns both the
    split table and the selected method name so the decision is recorded in
    output metadata.
    """
    candidates = entities.copy()
    if time_cols:
        usable = [column for column in time_cols if column in candidates.columns]
        if usable and candidates[usable].notna().any().any():
            split = make_forward_split(candidates, id_col, "time", usable, ratio)
            return split, "time"

    if pedigree_col and pedigree_col in candidates.columns:
        generations = candidates[pedigree_col].dropna()
        if generations.nunique() > 1:
            split = make_forward_split(candidates, id_col, "pedigree_generation", [pedigree_col], ratio)
            return split, "pedigree_generation"

    candidates["split_key_id"] = numeric_id_sort(candidates[id_col])
    split = make_forward_split(candidates, id_col, "id", ["split_key_id"], ratio)
    return split, "id"


def add_split(df: pd.DataFrame, split_df: pd.DataFrame) -> pd.DataFrame:
    """Attach split labels and forward order to any table with an ``ID`` column."""
    cols = ["ID", "split", "forward_order", "split_method"]
    return df.merge(split_df[cols], on="ID", how="left")


def melt_labels(
    df: pd.DataFrame,
    id_vars: list[str],
    value_vars: list[str],
    var_name: str,
    value_name: str,
) -> pd.DataFrame:
    """Convert a wide trait table into one row per animal-trait observation.

    Many modeling pipelines prefer long-form labels because different traits can
    be filtered, grouped, or evaluated independently. Missing trait values are
    dropped after melting.
    """
    long_df = df.melt(id_vars=id_vars, value_vars=value_vars, var_name=var_name, value_name=value_name)
    return long_df.dropna(subset=[value_name]).reset_index(drop=True)


def compute_pedigree_generation(pedigree: pd.DataFrame) -> pd.DataFrame:
    """Infer a simple generation depth from sire/dam relationships.

    Founders with no known parents inside the pedigree get generation 0. An
    animal with known parents receives one plus the deepest known parent
    generation. This gives a forward-order proxy when no explicit time field is
    available, while guarding against accidental cycles in the pedigree graph.
    """
    ped = pedigree.copy()
    for column in ["ID", "SIRE", "DAM"]:
        ped[column] = ped[column].map(normalize_id)

    parents = {
        row.ID: (
            row.SIRE if row.SIRE not in ("", "0") else None,
            row.DAM if row.DAM not in ("", "0") else None,
        )
        for row in ped[["ID", "SIRE", "DAM"]].itertuples(index=False)
    }

    @lru_cache(maxsize=None)
    def generation(animal_id: str, trail: tuple[str, ...] = ()) -> int:
        """Recursively compute one animal's generation with memoization."""
        if animal_id not in parents or animal_id in trail:
            return 0
        parent_generations = []
        for parent_id in parents[animal_id]:
            if parent_id and parent_id in parents:
                parent_generations.append(generation(parent_id, trail + (animal_id,)))
        return max(parent_generations) + 1 if parent_generations else 0

    ped["generation"] = ped["ID"].map(generation).astype(int)
    return ped


def export_wide_genotypes_by_split(
    genotype_path: Path,
    delimiter: str,
    split_df: pd.DataFrame,
    out_dir: Path,
) -> dict[str, int]:
    """Stream a wide genotype matrix into train/valid/test genotype CSV files.

    This optional export is intentionally streaming because HZA and PIC genotype
    files are too large to casually load as DataFrames. Each input row is routed
    according to the already computed split map, preserving the original marker
    columns and normalizing only the first ``ID`` field.
    """
    split_map = dict(zip(split_df["ID"], split_df["split"]))
    counts = {label: 0 for label in SPLIT_LABELS}
    writers = {}
    handles = {}
    try:
        for label in SPLIT_LABELS:
            out_path = out_dir / f"genotypes_{label}.csv"
            handles[label] = out_path.open("w", encoding="utf-8", newline="")
            writers[label] = csv.writer(handles[label])

        with genotype_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            header = handle.readline().strip()
            header_tokens = header.split(",") if delimiter == "," else header.split()
            header_tokens[0] = "ID"
            for writer in writers.values():
                writer.writerow(header_tokens)

            for line in handle:
                if not line.strip():
                    continue
                tokens = line.rstrip("\n\r").split(",") if delimiter == "," else line.split()
                animal_id = normalize_id(tokens[0])
                label = split_map.get(animal_id)
                if label:
                    tokens[0] = animal_id
                    writers[label].writerow(tokens)
                    counts[label] += 1
    finally:
        for handle in handles.values():
            handle.close()
    return counts


def read_plink_markers(map_path: Path) -> pd.DataFrame:
    """Read a PLINK MAP file into standard marker metadata columns."""
    markers = pd.read_csv(
        map_path,
        sep=r"\s+",
        header=None,
        names=["chromosome", "marker_id", "genetic_distance", "position"],
        dtype=str,
    )
    markers["position"] = pd.to_numeric(markers["position"], errors="coerce")
    markers["genetic_distance"] = pd.to_numeric(markers["genetic_distance"], errors="coerce")
    return markers


def export_multiple_plink_ped_dosages_by_split(
    ped_paths: list[Path],
    marker_ids: list[str],
    split_df: pd.DataFrame,
    out_dir: Path,
) -> dict[str, int]:
    """Convert multiple PLINK PED files into one split dosage matrix set.

    BloodLipid stores the three pig populations in separate PED files but with
    identical marker order. Dosage alleles are chosen from pooled allele counts
    across all included populations so the exported 0/1/2 coding is consistent
    for every sample.
    """
    allele_counts = [Counter() for _ in marker_ids]
    n_markers = len(marker_ids)

    for ped_path in ped_paths:
        with ped_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                tokens = line.split()
                alleles = tokens[6:]
                if len(alleles) < 2 * n_markers:
                    continue
                for idx in range(n_markers):
                    a1, a2 = alleles[2 * idx], alleles[2 * idx + 1]
                    if a1 != "0":
                        allele_counts[idx][a1] += 1
                    if a2 != "0":
                        allele_counts[idx][a2] += 1

    coding_rows = []
    alt_alleles = []
    for marker_id, counts in zip(marker_ids, allele_counts):
        if counts:
            ordered = sorted(counts.items(), key=lambda item: (item[1], item[0]))
            alt = ordered[0][0]
        else:
            alt = ""
        alt_alleles.append(alt)
        coding_rows.append(
            {
                "marker_id": marker_id,
                "dosage_allele": alt,
                "observed_alleles": ";".join(f"{allele}:{count}" for allele, count in sorted(counts.items())),
            }
        )
    write_csv(pd.DataFrame(coding_rows), out_dir / "plink_dosage_allele_coding.csv")

    split_map = dict(zip(split_df["ID"], split_df["split"]))
    counts = {label: 0 for label in SPLIT_LABELS}
    writers = {}
    handles = {}
    try:
        for label in SPLIT_LABELS:
            out_path = out_dir / f"genotypes_{label}.csv"
            handles[label] = out_path.open("w", encoding="utf-8", newline="")
            writers[label] = csv.writer(handles[label])
            writers[label].writerow(["ID"] + marker_ids)

        for ped_path in ped_paths:
            with ped_path.open("r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    tokens = line.split()
                    if len(tokens) < 6:
                        continue
                    animal_id = normalize_id(tokens[1])
                    label = split_map.get(animal_id)
                    if not label:
                        continue
                    alleles = tokens[6:]
                    if len(alleles) < 2 * n_markers:
                        continue
                    row = [animal_id]
                    for idx, alt in enumerate(alt_alleles):
                        a1, a2 = alleles[2 * idx], alleles[2 * idx + 1]
                        if not alt or a1 == "0" or a2 == "0":
                            row.append("")
                        else:
                            row.append(str(int(a1 == alt) + int(a2 == alt)))
                    writers[label].writerow(row)
                    counts[label] += 1
    finally:
        for handle in handles.values():
            handle.close()
    return counts


def process_hza(raw_root: Path, out_root: Path, ratio: tuple[float, float, float], export_genotypes: bool) -> dict:
    """Preprocess the HZA dataset and write standardized processed outputs.

    HZA has a whitespace-delimited wide genotype matrix, a marker map, phenotype
    tables, and mortality EBV files. There is no usable time or pedigree field,
    so the forward split falls back to sorted animal ID order after intersecting
    phenotype IDs with genotyped sample IDs.
    """
    raw = raw_root / "HZA"
    out = out_root / "HZA"
    out.mkdir(parents=True, exist_ok=True)

    genotype_ids = read_wide_genotype_ids(raw / "genotype.txt", delimiter="whitespace")

    phenotype = pd.read_csv(raw / "phenotype.txt", sep=r"\s+", na_values=MISSING_VALUES)
    phenotype = normalize_id_column(phenotype, "ID")
    phenotype = coerce_numeric(phenotype)
    phenotype = phenotype[phenotype["ID"].isin(set(genotype_ids))].copy()

    entities = phenotype[["ID"]].drop_duplicates()
    split_df, method = select_forward_split(entities, "ID", ratio)

    phenotype_split = add_split(phenotype, split_df)
    write_csv(phenotype_split, out / "phenotype_wide.csv")

    trait_cols = [column for column in phenotype.columns if column != "ID"]
    labels_long = melt_labels(phenotype_split, ["ID", "split", "forward_order", "split_method"], trait_cols, "trait", "value")
    write_csv(labels_long, out / "labels_long.csv")

    phenotype_csv = pd.read_csv(raw / "phenotype.csv", na_values=MISSING_VALUES)
    phenotype_csv = normalize_id_column(phenotype_csv, "ID")
    phenotype_csv["EBV"] = pd.to_numeric(phenotype_csv["EBV"], errors="coerce")
    phenotype_csv = add_split(phenotype_csv, split_df)
    write_csv(phenotype_csv.dropna(subset=["EBV"]), out / "phenotype_csv_long.csv")

    ebv_raw = pd.read_csv(raw / "mortality_EBV.txt", sep=r"\s+", na_values=MISSING_VALUES)
    ebv_rows = []
    columns = list(ebv_raw.columns)
    for idx in range(0, len(columns) - 1, 2):
        id_col, value_col = columns[idx], columns[idx + 1]
        trait = value_col.replace(".ebv", "")
        tmp = ebv_raw[[id_col, value_col]].rename(columns={id_col: "ID", value_col: "EBV"})
        tmp["ID"] = tmp["ID"].map(normalize_id)
        tmp["trait"] = trait
        ebv_rows.append(tmp[["ID", "trait", "EBV"]])
    mortality_ebv = pd.concat(ebv_rows, ignore_index=True)
    mortality_ebv["EBV"] = pd.to_numeric(mortality_ebv["EBV"], errors="coerce")
    mortality_ebv = add_split(mortality_ebv.dropna(subset=["EBV"]), split_df)
    write_csv(mortality_ebv, out / "mortality_ebv_long.csv")

    markers = pd.read_csv(raw / "map.txt", sep=r"\s+", dtype={"SNP": str})
    markers = markers.rename(columns={"SNP": "marker_id", "Chromosome": "chromosome", "Position": "position"})
    write_csv(markers, out / "markers.csv")

    samples = pd.DataFrame({"ID": genotype_ids})
    samples = add_split(samples, split_df)
    write_csv(samples, out / "genotype_samples.csv")
    write_csv(split_df, out / "splits.csv")

    exported_counts = None
    if export_genotypes:
        exported_counts = export_wide_genotypes_by_split(raw / "genotype.txt", "whitespace", split_df, out)

    summary = {
        "dataset": "HZA",
        "split_method": method,
        "split_ratio": list(ratio),
        "n_genotype_samples": len(genotype_ids),
        "n_phenotype_samples": int(phenotype["ID"].nunique()),
        "n_markers": int(len(markers)),
        "split_counts": split_summary(split_df),
        "exported_genotype_counts": exported_counts,
    }
    write_json(summary, out / "summary.json")
    return summary


def process_pic(raw_root: Path, out_root: Path, ratio: tuple[float, float, float], export_genotypes: bool) -> dict:
    """Preprocess the PIC dataset and write standardized processed outputs.

    PIC includes phenotype, EBV/accuracy, genotype, and pedigree files. Because
    the pedigree has useful parent links, the split is based on inferred
    pedigree generation, which better approximates forward validation than a
    random split or plain ID order.
    """
    raw = raw_root / "PIC"
    out = out_root / "PIC"
    out.mkdir(parents=True, exist_ok=True)

    genotype_ids = read_wide_genotype_ids(raw / "genotypes.txt", delimiter=",")
    genotype_id_set = set(genotype_ids)

    phenotype = pd.read_csv(raw / "phenotypes.txt", na_values=MISSING_VALUES)
    phenotype = normalize_id_column(phenotype, "ID")
    phenotype = coerce_numeric(phenotype)
    phenotype = phenotype[phenotype["ID"].isin(genotype_id_set)].copy()

    pedigree = pd.read_csv(raw / "pedigree.txt", na_values=MISSING_VALUES)
    pedigree = compute_pedigree_generation(pedigree)
    pedigree["is_genotyped"] = pedigree["ID"].isin(genotype_id_set)

    entities = phenotype[["ID"]].drop_duplicates().merge(pedigree[["ID", "generation"]], on="ID", how="left")
    split_df, method = select_forward_split(entities, "ID", ratio, pedigree_col="generation")

    phenotype_split = add_split(phenotype, split_df)
    write_csv(phenotype_split, out / "phenotype_wide.csv")
    trait_cols = [column for column in phenotype.columns if column != "ID"]
    labels_long = melt_labels(phenotype_split, ["ID", "split", "forward_order", "split_method"], trait_cols, "trait", "value")
    write_csv(labels_long, out / "labels_long.csv")

    ebvs = pd.read_csv(raw / "ebvs.txt", na_values=MISSING_VALUES)
    ebvs = ebvs.rename(columns={"Id": "ID"})
    ebvs = normalize_id_column(ebvs, "ID")
    ebvs = coerce_numeric(ebvs)
    ebvs = ebvs[ebvs["ID"].isin(genotype_id_set)].copy()
    ebvs_split = add_split(ebvs, split_df)
    write_csv(ebvs_split, out / "ebvs_wide.csv")

    ebv_cols = [column for column in ebvs.columns if column.lower().startswith("ebv")]
    acc_cols = [column for column in ebvs.columns if column.lower().startswith("acc")]
    if ebv_cols:
        ebv_long = melt_labels(ebvs_split, ["ID", "split", "forward_order", "split_method"], ebv_cols, "ebv_trait", "EBV")
        ebv_long["trait"] = ebv_long["ebv_trait"].str.replace("ebv", "t", regex=False)
        write_csv(ebv_long[["ID", "split", "forward_order", "split_method", "trait", "EBV"]], out / "ebv_long.csv")
    if acc_cols:
        acc_long = melt_labels(ebvs_split, ["ID", "split", "forward_order", "split_method"], acc_cols, "acc_trait", "accuracy")
        acc_long["trait"] = acc_long["acc_trait"].str.replace("acc", "t", regex=False)
        write_csv(acc_long[["ID", "split", "forward_order", "split_method", "trait", "accuracy"]], out / "accuracy_long.csv")

    pedigree_split = pedigree.merge(split_df[["ID", "split", "forward_order"]], on="ID", how="left")
    write_csv(pedigree_split, out / "pedigree.csv")

    markers = pd.DataFrame({"marker_id": read_wide_genotype_header(raw / "genotypes.txt", delimiter=",")[1:]})
    write_csv(markers, out / "markers.csv")

    samples = pd.DataFrame({"ID": genotype_ids})
    samples = add_split(samples, split_df)
    write_csv(samples, out / "genotype_samples.csv")
    write_csv(split_df, out / "splits.csv")

    exported_counts = None
    if export_genotypes:
        exported_counts = export_wide_genotypes_by_split(raw / "genotypes.txt", ",", split_df, out)

    summary = {
        "dataset": "PIC",
        "split_method": method,
        "split_ratio": list(ratio),
        "n_genotype_samples": len(genotype_ids),
        "n_phenotype_samples": int(phenotype["ID"].nunique()),
        "n_pedigree_animals": int(len(pedigree)),
        "n_markers": int(len(markers)),
        "split_counts": split_summary(split_df),
        "exported_genotype_counts": exported_counts,
    }
    write_json(summary, out / "summary.json")
    return summary


def process_bloodlipid(raw_root: Path, out_root: Path, ratio: tuple[float, float, float], export_genotypes: bool) -> dict:
    """Preprocess the BloodLipid pig GWAS dataset and write standardized outputs.

    The Dryad 4gh70 dataset contains three populations (DLY, EHL, and Laiwu),
    each with PLINK PED/MAP genotypes and six blood-lipid phenotypes. The
    phenotype files include experimental batch, so the animal-level split uses
    batch as a forward-validation proxy and keeps all traits for an animal in
    the same train/validation/test partition.
    """
    raw = raw_root / "BloodLipid" / "Primary_data" / "Primary_data" / "GWAS"
    out = out_root / "BloodLipid"
    out.mkdir(parents=True, exist_ok=True)

    populations = ["DLY", "EHL", "Laiwu"]
    trait_rename = {"HDL.C": "HDL-C", "LDL.C": "LDL-C"}
    phenotype_rows = []
    sample_rows = []
    ped_paths = []

    for population in populations:
        pheno_path = raw / f"{population}_60K_phens.txt"
        ped_path = raw / f"{population}_60K_gens.ped"
        ped_paths.append(ped_path)

        phenotype = pd.read_csv(pheno_path, sep=r"\s+", na_values=MISSING_VALUES)
        phenotype = phenotype.rename(columns={"id": "ID", **trait_rename})
        phenotype["ID"] = phenotype["ID"].map(normalize_id)
        phenotype["population"] = population
        phenotype_rows.append(phenotype)

        prefix = read_plink_ped_prefix(ped_path)
        prefix["population"] = population
        sample_rows.append(prefix)

    phenotype_wide = pd.concat(phenotype_rows, ignore_index=True)
    genotype_samples = pd.concat(sample_rows, ignore_index=True)
    genotype_id_set = set(genotype_samples["ID"])

    phenotype_wide = phenotype_wide[phenotype_wide["ID"].isin(genotype_id_set)].copy()
    for column in ["batch", "TCHOL", "TG", "HDL-C", "LDL-C", "HDL-C/LDL-C", "AI"]:
        if column in phenotype_wide.columns:
            phenotype_wide[column] = pd.to_numeric(phenotype_wide[column], errors="coerce")

    entities = (
        phenotype_wide[["ID", "population", "batch"]]
        .drop_duplicates("ID")
        .rename(columns={"batch": "split_key_batch"})
    )
    split_parts = []
    for population, group in entities.groupby("population", sort=True):
        group_split, _ = select_forward_split(group, "ID", ratio, time_cols=["split_key_batch"])
        group_split["split_method"] = "population_batch"
        split_parts.append(group_split)
    split_df = pd.concat(split_parts, ignore_index=True)
    method = "population_batch"

    phenotype_split = add_split(phenotype_wide, split_df)
    write_csv(phenotype_split, out / "phenotype_wide.csv")

    trait_cols = ["TCHOL", "TG", "HDL-C", "LDL-C", "HDL-C/LDL-C", "AI"]
    labels_long = melt_labels(
        phenotype_split,
        ["ID", "population", "sex", "batch", "split", "forward_order", "split_method"],
        trait_cols,
        "trait",
        "value",
    )
    write_csv(labels_long, out / "labels_long.csv")

    samples = genotype_samples.merge(split_df[["ID", "split", "forward_order", "split_method"]], on="ID", how="left")
    write_csv(samples, out / "genotype_samples.csv")
    write_csv(split_df, out / "splits.csv")

    markers = read_plink_markers(raw / "DLY_60K_gens.map")
    write_csv(markers, out / "markers.csv")

    exported_counts = None
    if export_genotypes:
        exported_counts = export_multiple_plink_ped_dosages_by_split(
            ped_paths,
            markers["marker_id"].astype(str).tolist(),
            split_df,
            out,
        )

    summary = {
        "dataset": "BloodLipid",
        "split_method": method,
        "split_ratio": list(ratio),
        "populations": populations,
        "n_genotype_samples": int(genotype_samples["ID"].nunique()),
        "n_phenotype_samples": int(phenotype_wide["ID"].nunique()),
        "n_markers": int(len(markers)),
        "n_traits": len(trait_cols),
        "split_counts": split_summary(split_df),
        "exported_genotype_counts": exported_counts,
    }
    write_json(summary, out / "summary.json")
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for raw paths, output paths, datasets, and ratio."""
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess HZA, PIC, and BloodLipid raw pig datasets and create 7:1:2 "
            "forward train/valid/test splits."
        )
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["HZA", "PIC", "BloodLipid"],
        choices=["HZA", "PIC", "BloodLipid"],
    )
    parser.add_argument("--ratio", nargs=3, type=float, default=(7.0, 1.0, 2.0), metavar=("TRAIN", "VALID", "TEST"))
    parser.add_argument(
        "--export-genotypes",
        action="store_true",
        help="Also export split genotype matrices. This is slow and writes large files.",
    )
    return parser.parse_args()


def main() -> None:
    """Run preprocessing for the selected datasets and write a combined summary."""
    args = parse_args()
    ratio = tuple(args.ratio)
    processors = {
        "HZA": process_hza,
        "PIC": process_pic,
        "BloodLipid": process_bloodlipid,
    }

    summaries = []
    for dataset in args.datasets:
        summaries.append(processors[dataset](args.raw_dir, args.out_dir, ratio, args.export_genotypes))

    write_json(
        {
            "raw_dir": str(args.raw_dir),
            "out_dir": str(args.out_dir),
            "datasets": summaries,
        },
        args.out_dir / "preprocess_summary.json",
    )
    for summary in summaries:
        counts = summary["split_counts"]
        print(
            f"{summary['dataset']}: method={summary['split_method']} "
            f"train={counts['train']} valid={counts['valid']} test={counts['test']}"
        )


if __name__ == "__main__":
    main()
