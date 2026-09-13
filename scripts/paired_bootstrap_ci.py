"""Paired bootstrap confidence intervals for EGT vs the strongest baseline.

The Round 1 response letter commits to testing model differences "using paired
bootstrap confidence intervals" on the same test individuals. This script
delivers that evidence.

Two comparisons are produced per dataset x trait cell:

``test_best``
    EGT (better of single/mtl by test PCC) against the strongest baseline by
    test PCC. This is the claim the original manuscript made -- "EGT beats the
    best competing method" -- so it is the comparison the reviewers asked to be
    tested with uncertainty, and it is deliberately generous to the baseline by
    letting the baseline be chosen on the test set.

``validation_locked``
    The pair recorded in ``selection_lock.json``: the EGT variant and the
    baseline that won on the seed-42 *validation* set. No test information is
    used to choose either side, so this variant is immune to the data-snooping
    objection even though it is not the most favourable comparison.

Protocol
--------
* Per-animal prediction = mean across the five replicate seeds (fixed-seed
  ensembling); per-seed differences are written separately so replicate
  variability stays visible.
* The paired bootstrap resamples the *same* animals for both models
  (B = 10,000), which is what makes the test paired. Reported interval is the
  percentile interval; the p-value is two-sided from the bootstrap standard
  error of the difference.
* Bias and regression slope are bootstrapped on the identical resamples so the
  "ranking versus calibration" claim is supported by intervals too.

Run from the project root:
    python scripts/paired_bootstrap_ci.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from animals_round1 import OUT, TRAITS  # noqa: E402

B = 10_000
SEEDS = (11, 17, 23, 29, 53)
BASELINES = ("gblup", "svr", "rf", "xgboost", "cnn", "rnn")
EGT_MODELS = ("single", "mtl")


def load_cell(dataset: str, trait: str) -> dict[str, list[pd.DataFrame]]:
    """Return {model: [per-seed test frames]} for one dataset/trait cell.

    Directory keys have the form ``{dataset}_{trait}_{model}_{seed}``, where
    ``trait`` is the literal token ``all`` for the multitrait EGT. Test animals
    are sorted by ID so every model and seed aligns row-for-row.
    """
    per_model: dict[str, list[pd.DataFrame]] = {}
    for seed in SEEDS:
        for directory in sorted((OUT / "repeat").glob(f"{dataset}_*_{seed}")):
            parts = directory.name.split("_")
            if len(parts) < 4 or parts[0] != dataset or parts[-1] != str(seed):
                continue
            model = "_".join(parts[2:-1])
            if model not in BASELINES + EGT_MODELS:
                continue
            frame = pd.read_csv(directory / "predictions.csv")
            frame = frame[(frame["split"] == "test") & (frame["trait"] == trait)]
            if not len(frame):
                continue
            frame = frame[["ID", "y_true", "y_pred"]].sort_values("ID").reset_index(drop=True)
            per_model.setdefault(model, []).append(frame)
    return per_model


def pcc(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    den = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else float("nan")


def slope_of(pred: np.ndarray, true: np.ndarray) -> float:
    p, t = pred - pred.mean(), true - true.mean()
    var = float((p * p).mean())
    return float((p * t).mean() / var) if var > 0 else float("nan")


def bootstrap_plan(n: int, rng: np.random.Generator) -> tuple[list[np.ndarray], np.ndarray]:
    """Draw the replicate plan for the paired, seed-resampled bootstrap.

    Returns ``(animal_index, seed_index)``. ``animal_index[b]`` holds the ``n``
    resampled animal positions for replicate ``b`` and ``seed_index[b]`` holds
    the ``len(SEEDS)`` replicate runs drawn with replacement for that replicate.
    For each resampled animal the pooled prediction is the mean of its
    prediction across the reselected seeds.
    """
    animal_idx = [rng.integers(0, n, size=n) for _ in range(B)]
    seed_idx = rng.integers(0, len(SEEDS), size=(B, len(SEEDS)))
    return animal_idx, seed_idx


def bootstrap_diffs(
    y: np.ndarray,
    egt_stack: np.ndarray,
    base_stack: np.ndarray,
    animal_idx: list[np.ndarray],
    seed_idx: np.ndarray,
) -> dict[str, np.ndarray]:
    """Bootstrap the EGT-minus-baseline difference in PCC, bias and slope.

    For every replicate the prediction of an animal is the mean of that animal's
    prediction over the drawn seeds, and correlation, bias and slope are all
    evaluated on the resampled animal set -- so each statistic is computed
    within the replicate rather than by reweighting a precomputed value.
    """
    y_mean = y.mean()
    pcc_d, bias_d, slope_d = [], [], []

    for a_idx, s_draw in zip(animal_idx, seed_idx):
        yy = y[a_idx]
        # index the drawn seeds for the drawn animals, then average across the
        # drawn seeds: shape (n_seeds_drawn, n) -> (n,)
        pe = egt_stack[s_draw][:, a_idx].mean(axis=0)
        pb = base_stack[s_draw][:, a_idx].mean(axis=0)

        def stats(pred: np.ndarray) -> tuple[float, float, float]:
            pc = pred - pred.mean()
            yc = yy - yy.mean()
            den = math.sqrt(float((pc * pc).sum()) * float((yc * yc).sum()))
            pcc_v = float((pc * yc).sum() / den) if den > 0 else float("nan")
            bias_v = float(pred.mean() - yy.mean())
            var = float((pc * pc).mean())
            slope_v = float((pc * yc).mean() / var) if var > 0 else float("nan")
            return pcc_v, bias_v, slope_v

        e_pcc, e_bias, e_slope = stats(pe)
        b_pcc, b_bias, b_slope = stats(pb)
        pcc_d.append(e_pcc - b_pcc)
        bias_d.append(e_bias - b_bias)
        slope_d.append(e_slope - b_slope)

    return {
        "pcc_diff": np.asarray(pcc_d),
        "bias_diff": np.asarray(bias_d),
        "slope_diff": np.asarray(slope_d),
    }


def describe(sample: np.ndarray, point: float) -> dict[str, float]:
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return {"point": point, "lo": float("nan"), "hi": float("nan"), "se": float("nan"), "p": float("nan")}
    se = float(sample.std(ddof=1))
    lo, hi = (float(v) for v in np.percentile(sample, [2.5, 97.5]))
    if se > 0:
        z = abs(point) / se
        p = float(2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))))
    else:
        p = float("nan")
    return {"point": float(point), "lo": lo, "hi": hi, "se": se, "p": p}


def best_by_test_pcc(per_model: dict[str, list[pd.DataFrame]], y: np.ndarray, candidates: tuple[str, ...]) -> str | None:
    scored = []
    for model in candidates:
        frames = per_model.get(model)
        if not frames or len(frames) != len(SEEDS):
            continue
        pred = np.mean([f["y_pred"].to_numpy(dtype=np.float64) for f in frames], axis=0)
        if len(pred) != len(y):
            continue
        value = pcc(pred, y)
        if np.isfinite(value):
            scored.append((value, model))
    return max(scored)[1] if scored else None


def main() -> None:
    lock = json.loads((OUT / "selection_lock.json").read_text(encoding="utf-8"))
    locked = {(w["dataset"], w["trait"], w["family"]): w["model"] for w in lock["winners"]}
    rng = np.random.default_rng(20260912)

    rows: list[dict] = []
    per_seed_rows: list[dict] = []

    for dataset, traits in TRAITS.items():
        for trait in traits:
            per_model = load_cell(dataset, trait)
            if not per_model:
                print(f"  skip {dataset}/{trait}: no predictions found")
                continue
            y = per_model["gblup"][0]["y_true"].to_numpy(dtype=np.float64)
            n = len(y)

            pairs: dict[str, tuple[str, str]] = {}
            egt_best = best_by_test_pcc(per_model, y, EGT_MODELS)
            base_best = best_by_test_pcc(per_model, y, BASELINES)
            if egt_best and base_best:
                pairs["test_best"] = (egt_best, base_best)

            le, lb = locked.get((dataset, trait, "EGT")), locked.get((dataset, trait, "baseline"))
            if le in per_model and lb in per_model:
                if (le, lb) != pairs.get("test_best"):
                    pairs["validation_locked"] = (le, lb)

            for scheme, (egt_model, base_model) in pairs.items():
                egt_frames, base_frames = per_model[egt_model], per_model[base_model]
                if len(egt_frames) != len(SEEDS) or len(base_frames) != len(SEEDS):
                    print(f"  skip {dataset}/{trait}/{scheme}: incomplete replicate coverage")
                    continue
                if not all((f["ID"].to_numpy() == per_model["gblup"][0]["ID"].to_numpy()).all() for f in egt_frames + base_frames):
                    print(f"  skip {dataset}/{trait}/{scheme}: test animals differ between models")
                    continue

                egt_stack = np.vstack([f["y_pred"].to_numpy(dtype=np.float64) for f in egt_frames])
                base_stack = np.vstack([f["y_pred"].to_numpy(dtype=np.float64) for f in base_frames])
                p_egt = egt_stack.mean(axis=0)
                p_base = base_stack.mean(axis=0)

                point = {
                    "pcc_diff": pcc(p_egt, y) - pcc(p_base, y),
                    "bias_diff": float(p_egt.mean() - p_base.mean()),
                    "slope_diff": slope_of(p_egt, y) - slope_of(p_base, y),
                }
                animal_idx, seed_idx = bootstrap_plan(n, rng)
                boot = bootstrap_diffs(y, egt_stack, base_stack, animal_idx, seed_idx)

                row = {
                    "scheme": scheme, "dataset": dataset, "trait": trait, "n_test": n, "n_seeds": len(SEEDS),
                    "egt_model": egt_model, "baseline_model": base_model,
                    "pcc_egt": pcc(p_egt, y), "pcc_baseline": pcc(p_base, y),
                }
                for key, sample in boot.items():
                    stats = describe(sample, point[key])
                    row[f"{key}_point"] = stats["point"]
                    row[f"{key}_lo"] = stats["lo"]
                    row[f"{key}_hi"] = stats["hi"]
                    row[f"{key}_se"] = stats["se"]
                    row[f"{key}_p"] = stats["p"]
                row["pcc_significant_0_05"] = bool(row["pcc_diff_lo"] > 0 or row["pcc_diff_hi"] < 0)
                rows.append(row)

                for i in range(len(SEEDS)):
                    e = egt_frames[i]["y_pred"].to_numpy(dtype=np.float64)
                    b = base_frames[i]["y_pred"].to_numpy(dtype=np.float64)
                    per_seed_rows.append({
                        "scheme": scheme, "dataset": dataset, "trait": trait, "seed": SEEDS[i],
                        "egt_model": egt_model, "baseline_model": base_model,
                        "pcc_egt": pcc(e, y), "pcc_baseline": pcc(b, y), "pcc_diff": pcc(e, y) - pcc(b, y),
                    })

    summary = pd.DataFrame(rows)
    per_seed = pd.DataFrame(per_seed_rows)
    if summary.empty:
        raise SystemExit("No cell produced a complete paired comparison; nothing written.")

    summary.to_csv(OUT / "paired_bootstrap_ci.csv", index=False)
    per_seed.to_csv(OUT / "paired_bootstrap_per_seed.csv", index=False)

    agreement = (
        per_seed.assign(positive=lambda d: d["pcc_diff"] > 0)
        .groupby(["scheme", "dataset", "trait"])
        .positive.agg(["sum", "count"])
        .rename(columns={"sum": "n_seeds_egt_higher", "count": "n_seeds"})
        .reset_index()
    )
    agreement.to_csv(OUT / "paired_bootstrap_sign_agreement.csv", index=False)

    lines: list[str] = []
    for scheme in ("test_best", "validation_locked"):
        block = summary[summary.scheme == scheme]
        if block.empty:
            continue
        lines.append("=" * 108)
        title = ("PRIMARY: EGT vs strongest baseline (baseline chosen on test PCC)"
                 if scheme == "test_best"
                 else "SECONDARY: validation-locked pair (seed-42 validation winners; no test information)")
        lines.append(title)
        lines.append("=" * 108)
        lines.append("")
        lines.append(f"{'Dataset':<11} {'Trait':<7} {'n':>4} {'EGT':<7} {'vs base':<8} {'dPCC':>8} {'95% CI':>21} {'p':>9} {'sig':<4} {'seeds EGT higher'}")
        lines.append("-" * 108)
        for r in block.sort_values(["dataset", "trait"]).itertuples(index=False):
            a = agreement[(agreement.scheme == scheme) & (agreement.dataset == r.dataset) & (agreement.trait == r.trait)]
            agree = f"{int(a.n_seeds_egt_higher.iloc[0])}/{int(a.n_seeds.iloc[0])}" if len(a) else "-"
            lines.append(
                f"{r.dataset:<11} {r.trait:<7} {r.n_test:>4} {r.egt_model:<7} {r.baseline_model:<8} "
                f"{r.pcc_diff_point:>+8.4f} [{r.pcc_diff_lo:>+7.4f}, {r.pcc_diff_hi:>+7.4f}] "
                f"{r.pcc_diff_p:>9.4f} {'YES' if r.pcc_significant_0_05 else 'no':<4} {agree}"
            )
        lines.append("")
        n = len(block)
        lines.append(f"cells: {n}   EGT numerically higher: {int((block.pcc_diff_point > 0).sum())}/{n}   "
                     f"95% CI excludes zero: {int(block.pcc_significant_0_05.sum())}/{n}")
        lines.append("")
        lines.append(f"{'Dataset':<11} {'Trait':<7} {'dBias':>9} {'95% CI':>21}   {'dSlope':>8} {'95% CI':>21}")
        lines.append("-" * 108)
        for r in block.sort_values(["dataset", "trait"]).itertuples(index=False):
            lines.append(
                f"{r.dataset:<11} {r.trait:<7} {r.bias_diff_point:>+9.4f} "
                f"[{r.bias_diff_lo:>+7.4f}, {r.bias_diff_hi:>+7.4f}]   "
                f"{r.slope_diff_point:>+8.4f} [{r.slope_diff_lo:>+7.4f}, {r.slope_diff_hi:>+7.4f}]"
            )
        lines.append("")

    (OUT / "paired_bootstrap_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
