"""Population-structure test for the attributed SSC13 marker (reviewers R2C8, R1C3).

Two questions are answered here, both on the pooled BloodLipid cohort:

1. Does MARC0013088 dosage simply track ancestry? If the marker is a
   between-breed frequency difference rather than a within-population variant,
   its attribution score reflects population structure, not a lipid QTL.
2. Does the phenotype itself carry structure? A phenotype that differs between
   populations makes any ancestry-correlated marker look predictive.

The marker is monomorphic in Erhualian (see ssc13_local_ld.py), so the test
below is the direct one: correlate dosage with the leading principal components,
and correlate both dosage and phenotype with population label.

Genotypes are mean-imputed and scaled before PCA, matching the pipeline used for
model inputs. Coordinates are Sscrofa10.2.

Run from the project root:
    python scripts/ssc13_structure_test.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "BloodLipid"
OUT = ROOT / "results" / "animals_round1" / "ssc13_annotation"
FOCAL = "MARC0013088"
TRAITS = ["LDL-C", "TCHOL", "TG", "HDL-C"]


def load_genotypes() -> pd.DataFrame:
    frames = [pd.read_csv(DATA / f"genotypes_{s}.csv", index_col=0) for s in ("train", "valid", "test")]
    return pd.concat(frames, axis=0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    geno = load_genotypes()
    splits = pd.read_csv(DATA / "splits.csv", dtype={"ID": str})
    pop_map = dict(zip(splits["ID"], splits["population"])) if "population" in splits.columns else {}
    pops = np.array([pop_map.get(str(i), "unknown") for i in geno.index])
    labels = pd.read_csv(DATA / "labels_long.csv", dtype={"ID": str})

    print("=" * 80)
    print("Population-structure test for the attributed SSC13 marker")
    print("=" * 80)
    print(f"animals: {len(geno)}   markers: {geno.shape[1]}")

    # --- PCA on the pooled genotype matrix (mean-impute, then standardise) ---
    x = geno.to_numpy(dtype=np.float64)
    col_mean = np.nanmean(x, axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    nan_mask = ~np.isfinite(x)
    x[nan_mask] = np.take(col_mean, np.where(nan_mask)[1])
    col_sd = x.std(axis=0)
    keep = col_sd > 0
    x = (x[:, keep] - x[:, keep].mean(axis=0)) / col_sd[keep]
    # thin the matrix for a fast, stable decomposition
    rng = np.random.default_rng(11)
    subset = rng.choice(x.shape[1], size=min(40000, x.shape[1]), replace=False)
    u, s, vt = np.linalg.svd(x[:, subset], full_matrices=False)
    pcs = u[:, :10] * s[:10]
    var_explained = (s[:10] ** 2) / np.sum(s ** 2)
    print("\nvariance explained by PC1..PC10:")
    print("  " + "  ".join(f"PC{i+1}={v:.3f}" for i, v in enumerate(var_explained[:10])))

    pc_df = pd.DataFrame(pcs, columns=[f"PC{i+1}" for i in range(10)], index=geno.index)

    # --- test 1: does the focal marker track ancestry? ---
    dosage = geno[FOCAL].to_numpy(dtype=np.float64)
    maf = min(np.nanmean(dosage) / 2.0, 1 - np.nanmean(dosage) / 2.0)
    print(f"\n[{FOCAL}] MAF pooled: {maf:.4f}")
    print("\ncorrelation of MARC0013088 dosage with leading PCs:")
    corr_rows = []
    for pc in pc_df.columns:
        v = pc_df[pc].to_numpy(dtype=np.float64)
        mask = np.isfinite(dosage) & np.isfinite(v)
        r = float(np.corrcoef(dosage[mask], v[mask])[0, 1]) if mask.sum() > 10 and dosage[mask].std() > 0 else float("nan")
        corr_rows.append({"component": pc, "r_with_dosage": r, "r2": r * r if np.isfinite(r) else np.nan})
    corr_df = pd.DataFrame(corr_rows)
    print(corr_df.to_string(index=False))

    # --- test 2: is the marker frequency itself structured by population? ---
    print("\nfocal allele frequency by population:")
    freq_rows = []
    for pop in sorted(set(pops)):
        if pop == "unknown":
            continue
        d = dosage[pops == pop]
        f = float(np.nanmean(d) / 2.0)
        freq_rows.append({"population": pop, "n": int(d.size), "allele_freq": f,
                          "maf": min(f, 1 - f), "n_carriers": int(np.nansum(d > 0))})
    freq_df = pd.DataFrame(freq_rows)
    print(freq_df.to_string(index=False))

    # --- test 3: is the phenotype structured by population? ---
    print("\nmean phenotype by population (structure in the trait itself):")
    pheno_wide = pd.read_csv(DATA / "phenotype_wide.csv", dtype={"ID": str})
    if "ID" not in pheno_wide.columns:
        pheno_wide = pheno_wide.rename(columns={pheno_wide.columns[0]: "ID"})
    pheno_wide = pheno_wide.set_index("ID")
    available_traits = [t for t in TRAITS if t in pheno_wide.columns]
    pheno_wide = pheno_wide.assign(population=[pop_map.get(str(i), "unknown") for i in pheno_wide.index])
    trait_means = pheno_wide.groupby("population")[available_traits].mean()
    trait_sds = pheno_wide.groupby("population")[available_traits].std()
    print(trait_means.round(4).to_string())
    # between-population spread relative to within-population SD
    spread = (trait_means.max() - trait_means.min()) / trait_sds.mean()
    print("\n(between-population range) / (mean within-population SD):")
    for t, v in spread.items():
        print(f"  {t:<7} {v:.3f}")

    summary = {
        "focal_marker": FOCAL,
        "assembly": "Sscrofa10.2",
        "n_animals": int(len(geno)),
        "focal_maf_pooled": maf,
        "pc_variance_explained": {f"PC{i+1}": float(v) for i, v in enumerate(var_explained[:10])},
        "max_abs_r_dosage_vs_pc": float(corr_df.r_with_dosage.abs().max()),
        "pc_with_max_corr": str(corr_df.loc[corr_df.r_with_dosage.abs().idxmax(), "component"]),
        "focal_freq_by_population": freq_df.to_dict(orient="records"),
        "phenotype_mean_by_population": trait_means.round(6).to_dict(),
        "phenotype_between_over_within_sd": {k: float(v) for k, v in spread.items()},
    }
    (OUT / "ssc13_structure_test.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    corr_df.to_csv(OUT / "ssc13_dosage_vs_pc.csv", index=False)
    freq_df.to_csv(OUT / "ssc13_allele_freq_by_population.csv", index=False)
    trait_means.to_csv(OUT / "ssc13_phenotype_mean_by_population.csv")
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
