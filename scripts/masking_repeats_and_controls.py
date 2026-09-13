"""Repeated random masking with confidence intervals plus positive controls.

Reviewer 2 Comment 9 asks for two things the original Figure 7 lacked:

1. repeated random masking runs, so that the random-masking reference has a
   confidence interval rather than a single draw; and
2. a positive control, obtained by masking regions harbouring well-established
   porcine lipid-metabolism genes.

The published pipeline already supports repeated draws via ``--random-repeats``
but was run with its default of 1, which is why Figure 7 carries no interval.
This script reuses the checkpoint, the identical masking convention (masked
markers are set to the training mean, i.e. 0 in standardised space) and the
identical metric, and adds the missing replication and controls. No retraining
is involved.

Run from the project root:
    python scripts/masking_repeats_and_controls.py
    python scripts/masking_repeats_and_controls.py --explain-root results/bloodlipid_explainability_pccorr \
        --out results/animals_round1/masking_controls_pccorr
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from explain_egt import (  # noqa: E402
    config_from_checkpoint,
    load_checkpoint,
    load_single_trait_bundle,
    restore_model,
    transform_X,
)
from egt import evaluate_predictions, get_torch_device, predict_arrays  # noqa: E402

SRC_GENO = ROOT / "data" / "processed" / "BloodLipid"
EXPLAIN = ROOT / "results" / "bloodlipid_explainability_final"
EXPLAIN_ROOT = EXPLAIN
OUT = ROOT / "results" / "animals_round1" / "masking_controls"
TRAITS = ["LDL-C", "TCHOL", "TG", "HDL-C"]
TOP_FRACS = [0.01, 0.05, 0.10]
N_RANDOM = 20          # independent random-masking draws per (trait, fraction)
WINDOW_SIZE = 256
BASE_SEED = 20260912

# Positive controls: established porcine lipid / lipoprotein metabolism genes.
# The 60K chip gives only about one marker per 33 kb, so a gene-sized window holds
# just a handful of markers and is far too small to perturb the model. Controls are
# therefore defined at locus scale (5 Mb), and the number of markers masked is
# always reported so underpowered regions are obvious.
POSITIVE_CONTROLS = {
    "APOB_locus_5Mb":    ("3", 114_700_000, 119_700_000),
    "PCSK9_locus_5Mb":   ("4", 105_000_000, 110_000_000),
    "SSC13_candidate":   ("13", 138_000_000, 143_000_000),
    "SSC13_whole_chrom": ("13", 0, 300_000_000),
    "SSC3_whole_chrom":  ("3", 0, 400_000_000),
}
# The attributed marker's own window
ATTRIBUTED = ("13", 140_497_664, 140_497_920)


class Args:
    def __init__(self, trait: str) -> None:
        self.checkpoint = EXPLAIN_ROOT / "models" / trait / "best_checkpoint.pt"
        self.genotype_dir = SRC_GENO
        self.labels = SRC_GENO / "labels_long.csv"
        self.trait = trait
        self.dataset = "BloodLipid"
        self.target_column = "value"
        self.id_column = "ID"
        self.trait_column = "trait"
        self.split_column = "split"
        self.device = "cuda"
        self.batch_size = 32
        self.num_workers = 0
        self.ig_steps = 24
        self.window_size = WINDOW_SIZE


def markers_for_region(markers: pd.DataFrame, chrom: str, start: int, end: int) -> list[str]:
    mask = (
        (markers["window_chrom"].astype(str) == str(chrom))
        & (markers["window_start"] >= start)
        & (markers["window_end"] <= end)
    )
    return markers.loc[mask, "marker_id"].tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--explain-root", type=Path, default=EXPLAIN,
                    help="directory holding models/<trait> and explanations/<trait>")
    ap.add_argument("--out", type=Path, default=OUT)
    cli = ap.parse_args()
    global EXPLAIN_ROOT
    EXPLAIN_ROOT = cli.explain_root
    out_dir: Path = cli.out
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    region_rows: list[dict] = []

    for trait in TRAITS:
        args = Args(trait)
        if not args.checkpoint.exists():
            print(f"  skip {trait}: checkpoint missing")
            continue

        payload = load_checkpoint(args.checkpoint)
        config = config_from_checkpoint(payload, args)
        bundle = load_single_trait_bundle(args)
        model, device = restore_model(payload, config, len(bundle.marker_names))
        _xtr, _xval, X_test = transform_X(bundle, payload)
        y_means = np.asarray(payload["y_means"], dtype=np.float32)
        y_stds = np.asarray(payload["y_stds"], dtype=np.float32)
        marker_to_idx = {m: i for i, m in enumerate(bundle.marker_names)}

        base_pred = predict_arrays(model, X_test, config.batch_size, config.num_workers, device, config.amp)
        base = evaluate_predictions(bundle, bundle.test, base_pred, y_means, y_stds)[bundle.traits[0]]
        print(f"[{trait}] baseline test PCC = {base['pearson']:.4f}")

        windows = pd.read_csv(EXPLAIN_ROOT / "explanations" / trait / "window_importance.csv")
        windows = windows.sort_values("window_importance", ascending=False).reset_index(drop=True)
        # marker -> window key consistent with the pipeline (start coordinate)
        markers = pd.read_csv(EXPLAIN_ROOT / "explanations" / trait / "snp_importance.csv")
        markers["window_start"] = (markers["position"] // WINDOW_SIZE) * WINDOW_SIZE
        markers["window_end"] = markers["window_start"] + WINDOW_SIZE
        markers["window_key"] = markers["window_start"].astype(str)
        markers["window_chrom"] = markers["chromosome"]
        windows["window_key"] = windows["window_start"].astype(str)
        window_keys = windows["window_key"].tolist()
        n_windows = len(window_keys)

        def masked_pcc(marker_ids: list[str]) -> tuple[float, int]:
            idx = [marker_to_idx[m] for m in marker_ids if m in marker_to_idx]
            X = X_test.copy()
            if idx:
                X[:, idx] = 0.0
            pred = predict_arrays(model, X, config.batch_size, config.num_workers, device, config.amp)
            metrics = evaluate_predictions(bundle, bundle.test, pred, y_means, y_stds)[bundle.traits[0]]
            return float(metrics["pearson"]), len(idx)

        rng = np.random.default_rng(BASE_SEED)

        for frac in TOP_FRACS:
            k = max(1, int(round(n_windows * frac)))

            top_ids = markers[markers["window_key"].isin(set(window_keys[:k]))]["marker_id"].tolist()
            pcc_top, n_top = masked_pcc(top_ids)
            all_rows.append({"trait": trait, "mask_fraction": frac, "strategy": "top",
                             "repeat": 0, "n_windows": k, "n_markers_masked": n_top,
                             "baseline_PCC": float(base["pearson"]), "masked_PCC": pcc_top,
                             "delta_PCC": pcc_top - float(base["pearson"])})

            low_ids = markers[markers["window_key"].isin(set(window_keys[-k:]))]["marker_id"].tolist()
            pcc_low, n_low = masked_pcc(low_ids)
            all_rows.append({"trait": trait, "mask_fraction": frac, "strategy": "low",
                             "repeat": 0, "n_windows": k, "n_markers_masked": n_low,
                             "baseline_PCC": float(base["pearson"]), "masked_PCC": pcc_low,
                             "delta_PCC": pcc_low - float(base["pearson"])})

            for rep in range(N_RANDOM):
                keys = set(rng.choice(window_keys, size=k, replace=False).tolist())
                ids = markers[markers["window_key"].isin(keys)]["marker_id"].tolist()
                pcc_r, n_r = masked_pcc(ids)
                all_rows.append({"trait": trait, "mask_fraction": frac, "strategy": "random",
                                 "repeat": rep, "n_windows": k, "n_markers_masked": n_r,
                                 "baseline_PCC": float(base["pearson"]), "masked_PCC": pcc_r,
                                 "delta_PCC": pcc_r - float(base["pearson"])})

            # Matched-size control. The top-attributed windows are not a random
            # sample of windows: they are the densest ones, so masking them removes
            # far more markers than a random draw of the same window count. Without
            # holding masked-marker count fixed, "top is worse than random" cannot
            # be separated from "top removes more markers".
            n_top_markers = max(1, n_top)
            all_ids = markers["marker_id"].tolist()
            for rep in range(N_RANDOM):
                ids = rng.choice(all_ids, size=n_top_markers, replace=False).tolist()
                pcc_m, n_m = masked_pcc(ids)
                all_rows.append({"trait": trait, "mask_fraction": frac,
                                 "strategy": "random_matched_markers",
                                 "repeat": rep, "n_windows": n_top_markers, "n_markers_masked": n_m,
                                 "baseline_PCC": float(base["pearson"]), "masked_PCC": pcc_m,
                                 "delta_PCC": pcc_m - float(base["pearson"])})
            print(f"  frac={frac}: top {pcc_top - float(base['pearson']):+.4f} on {n_top} markers, "
                  f"low {pcc_low - float(base['pearson']):+.4f} on {n_low}, "
                  f"plus {N_RANDOM} window-random and {N_RANDOM} marker-matched draws")

        # positive controls and the attributed marker's own window
        for name, (chrom, start, end) in {**POSITIVE_CONTROLS, "MARC0013088_window": ATTRIBUTED}.items():
            ids = (markers_for_region(markers, chrom, start, end) if name != "MARC0013088_window"
                   else markers[markers["window_key"] == str(ATTRIBUTED[1])]["marker_id"].tolist())
            if not ids:
                continue
            pcc_v, n_v = masked_pcc(ids)
            region_rows.append({"trait": trait, "region": name, "chromosome": chrom,
                                "start": start, "end": end, "n_markers_masked": n_v,
                                "baseline_PCC": float(base["pearson"]), "masked_PCC": pcc_v,
                                "delta_PCC": pcc_v - float(base["pearson"])})
        print(f"  controls: {len([r for r in region_rows if r['trait']==trait])} regions masked")

    repeats = pd.DataFrame(all_rows)
    regions = pd.DataFrame(region_rows)
    repeats.to_csv(out_dir / "masking_repeats_long.csv", index=False)
    regions.to_csv(out_dir / "masking_positive_controls.csv", index=False)

    # summarise: top/low single values vs the random distributions
    summary_rows = []
    for (trait, frac), grp in repeats.groupby(["trait", "mask_fraction"]):
        rand = grp[grp.strategy == "random"]["delta_PCC"]
        matched = grp[grp.strategy == "random_matched_markers"]["delta_PCC"]
        top = grp[grp.strategy == "top"]["delta_PCC"]
        low = grp[grp.strategy == "low"]["delta_PCC"]
        if not len(rand):
            continue
        lo, hi = np.percentile(rand, [2.5, 97.5])
        mlo, mhi = np.percentile(matched, [2.5, 97.5]) if len(matched) else (np.nan, np.nan)
        top_v = float(top.iloc[0]) if len(top) else np.nan
        marker_counts = grp[grp.strategy == "top"]["n_markers_masked"]
        summary_rows.append({
            "trait": trait, "mask_fraction": frac,
            "n_markers_masked_top": int(marker_counts.iloc[0]) if len(marker_counts) else 0,
            "n_random_draws": len(rand),
            "random_mean_delta": float(rand.mean()), "random_sd_delta": float(rand.std(ddof=1)),
            "random_ci_lo": float(lo), "random_ci_hi": float(hi),
            "matched_mean_delta": float(matched.mean()) if len(matched) else np.nan,
            "matched_sd_delta": float(matched.std(ddof=1)) if len(matched) else np.nan,
            "matched_ci_lo": float(mlo), "matched_ci_hi": float(mhi),
            "top_delta": top_v,
            "top_outside_window_random_ci": bool(top_v < lo),
            "top_outside_matched_ci": bool(top_v < mlo),
            "low_delta": float(low.iloc[0]) if len(low) else np.nan,
            "top_minus_random_mean": top_v - float(rand.mean()),
            "top_minus_matched_mean": top_v - float(matched.mean()) if len(matched) else np.nan,
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "masking_repeats_summary.csv", index=False)

    lines = ["=" * 116,
             "Repeated random masking with confidence intervals (Reviewer 2, Comment 9)",
             "=" * 116, "",
             f"independent draws per cell: {N_RANDOM} (window-matched) and {N_RANDOM} (marker-matched)",
             "delta_PCC = masked PCC - baseline PCC; more negative = more damaging to prediction",
             "intervals are 2.5-97.5 percentiles of the corresponding random distribution", ""]
    lines.append(f"{'trait':<7} {'frac':>5} {'#mkr top':>9} {'top dPCC':>10} "
                 f"{'window-random mean [95% CI]':>34} {'marker-matched mean [95% CI]':>34} {'top worse than matched?':>24}")
    lines.append("-" * 116)
    for r in summary.itertuples(index=False):
        wci = f"{r.random_mean_delta:+.4f} [{r.random_ci_lo:+.4f}, {r.random_ci_hi:+.4f}]"
        mci = f"{r.matched_mean_delta:+.4f} [{r.matched_ci_lo:+.4f}, {r.matched_ci_hi:+.4f}]"
        lines.append(f"{r.trait:<7} {r.mask_fraction:>5} {r.n_markers_masked_top:>9} {r.top_delta:>+10.4f} "
                     f"{wci:>34} {mci:>34} {'YES' if r.top_outside_matched_ci else 'no':>24}")
    lines.append("")
    lines.append("Positive controls at locus scale (established pig lipid genes) and the attributed marker window:")
    lines.append("")
    lines.append(f"{'trait':<7} {'region':<22} {'chr':>4} {'n_masked':>9} {'delta_PCC':>11}")
    lines.append("-" * 116)
    for r in regions.sort_values(["trait", "delta_PCC"]).itertuples(index=False):
        lines.append(f"{r.trait:<7} {r.region:<22} {r.chromosome:>4} {r.n_markers_masked:>9} {r.delta_PCC:>+11.4f}")
    (out_dir / "masking_repeats_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
