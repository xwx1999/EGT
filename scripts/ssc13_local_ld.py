"""Local LD structure around the SSC13 candidate marker MARC0013088.

Added for the Animals Round 1 revision (reviewers R1C3, R2C8). The manuscript
reports a recurrent attribution peak at MARC0013088 and describes it as an SSC13
"hotspot". A hotspot claim implies linkage disequilibrium: a marker standing in
for a surrounding region. This script tests that directly by measuring r^2
between MARC0013088 and every marker within 1 Mb on either side, in the pooled
BloodLipid cohort and within each of the three component populations.

Coordinates are Sscrofa10.2 (see assembly_determination.md): the marker maps to
SSC13:140,497,749 on that build, and the associated marker map uses chromosome 0
for unplaced markers.

Run from the project root:
    python scripts/ssc13_local_ld.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "BloodLipid"
OUT = ROOT / "results" / "animals_round1" / "ssc13_annotation"
FOCAL = "MARC0013088"
WINDOW = 1_000_000  # +/- 1 Mb


def load_genotypes() -> pd.DataFrame:
    frames = []
    for split in ("train", "valid", "test"):
        frame = pd.read_csv(DATA / f"genotypes_{split}.csv", index_col=0)
        frames.append(frame)
    pooled = pd.concat(frames, axis=0)
    return pooled


def allele_freq(series: pd.Series) -> float:
    values = series.dropna().to_numpy(dtype=float)
    return float(values.mean() / 2.0) if values.size else float("nan")


def r_squared(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 10:
        return float("nan")
    x, y = a[mask], b[mask]
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1] ** 2)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    markers = pd.read_csv(DATA / "markers.csv")
    markers["chromosome"] = markers["chromosome"].astype(str)
    markers = markers.drop_duplicates("marker_id").set_index("marker_id")

    focal_rows = markers[markers.index == FOCAL]
    if focal_rows.empty:
        raise SystemExit(f"{FOCAL} not present in the BloodLipid marker map.")
    focal = focal_rows.iloc[0]
    chrom, pos = str(focal["chromosome"]), int(focal["position"])
    print(f"focal marker : {FOCAL}  chr{chrom}:{pos:,} (Sscrofa10.2)")

    on_chrom = markers[(markers["chromosome"] == chrom)].copy()
    neighbours = on_chrom[(on_chrom["position"] - pos).abs() <= WINDOW].copy()
    neighbour_ids = [m for m in neighbours.index if m in set(pd.read_csv(DATA / "genotypes_test.csv", nrows=1).columns)]
    neighbour_ids = sorted(neighbour_ids, key=lambda m: int(markers.loc[m, "position"]))
    print(f"markers within +/-{WINDOW/1e6:.1f} Mb on chr{chrom}: {len(neighbour_ids)}")

    pooled = load_genotypes()
    splits = pd.read_csv(DATA / "splits.csv", dtype={"ID": str})
    if "population" in splits.columns:
        pop_map = dict(zip(splits["ID"], splits["population"]))
    else:
        pop_map = {}

    available = [m for m in neighbour_ids if m in pooled.columns]
    matrix = pooled[available]
    ids = pooled.index.to_numpy()
    populations = np.array([pop_map.get(str(i), "unknown") for i in ids])
    print("populations present:", {p: int((populations == p).sum()) for p in np.unique(populations)})

    rows = []
    for marker in available:
        if marker == FOCAL:
            continue
        record = {"marker_id": marker, "chromosome": chrom, "position": int(markers.loc[marker, "position"])}
        record["distance_bp"] = record["position"] - pos
        record["maf_pooled"] = min(allele_freq(matrix[marker]), 1 - allele_freq(matrix[marker]))
        record["r2_pooled"] = r_squared(pooled[FOCAL].to_numpy(dtype=float), matrix[marker].to_numpy(dtype=float))
        for pop in sorted(set(populations)):
            if pop == "unknown":
                continue
            sub = pooled.loc[populations == pop]
            record[f"maf_{pop}"] = min(allele_freq(sub[marker]), 1 - allele_freq(sub[marker]))
            record[f"r2_{pop}"] = r_squared(sub[FOCAL].to_numpy(dtype=float), sub[marker].to_numpy(dtype=float))
        rows.append(record)

    ld = pd.DataFrame(rows).sort_values("position").reset_index(drop=True)
    ld.to_csv(OUT / "ssc13_local_ld.csv", index=False)

    focal_maf = min(allele_freq(pooled[FOCAL]), 1 - allele_freq(pooled[FOCAL]))
    summary = {
        "focal_marker": FOCAL,
        "assembly": "Sscrofa10.2",
        "coordinate": f"SSC13:{pos}",
        "n_animals_pooled": int(len(pooled)),
        "n_neighbour_markers": int(len(ld)),
        "window_bp": WINDOW,
        "focal_maf_pooled": focal_maf,
        "focal_maf_by_population": {
            p: min(allele_freq(pooled.loc[populations == p, FOCAL]), 1 - allele_freq(pooled.loc[populations == p, FOCAL]))
            for p in sorted(set(populations)) if p != "unknown"
        },
        "max_r2_pooled": float(np.nanmax(ld["r2_pooled"])) if len(ld) else None,
        "n_markers_r2_gt_0.2": int((ld["r2_pooled"] > 0.2).sum()) if len(ld) else 0,
        "n_markers_r2_gt_0.5": int((ld["r2_pooled"] > 0.5).sum()) if len(ld) else 0,
        "median_r2_pooled": float(np.nanmedian(ld["r2_pooled"])) if len(ld) else None,
    }
    (OUT / "ssc13_local_ld_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print("=" * 78)
    print("LD around MARC0013088")
    print("=" * 78)
    print(f"animals (pooled)          : {summary['n_animals_pooled']}")
    print(f"focal MAF (pooled)        : {focal_maf:.4f}")
    for p, v in summary["focal_maf_by_population"].items():
        print(f"focal MAF ({p:<12}) : {v:.4f}")
    print(f"median r^2 to neighbours  : {summary['median_r2_pooled']:.4f}")
    print(f"max r^2 to a neighbour    : {summary['max_r2_pooled']:.4f}")
    print(f"neighbours with r^2 > 0.2 : {summary['n_markers_r2_gt_0.2']}")
    print(f"neighbours with r^2 > 0.5 : {summary['n_markers_r2_gt_0.5']}")
    print()
    top = ld.sort_values("r2_pooled", ascending=False).head(12)
    print("strongest LD partners:")
    print(top[["marker_id", "position", "distance_bp", "maf_pooled", "r2_pooled"]].to_string(index=False))
    print()
    print(f"written: {OUT / 'ssc13_local_ld.csv'}")


if __name__ == "__main__":
    main()
