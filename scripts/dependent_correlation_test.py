"""Formal test for dependent correlations, on a common test set.

Reviewer 2 Comment 3 asks for paired bootstrap confidence intervals "or formal
tests for dependent correlations". The paired bootstrap is already computed
(``paired_bootstrap_ci.py``); this script adds the classic test for the
difference between two correlations that share one variable, so the response can
cite both.

The test is Williams' t (Williams 1959; Steiger 1980): for correlations r_jk and
r_jh measured on the same n individuals, the two are dependent, and the naive
independent-samples comparison is invalid. The statistic is

    t = (r_jk - r_hk) * sqrt( ((n - 1) * (1 + r_jh)) / (2 * ((n-1)/(n-3)) * det(R) + rbar^2 * (1 - r_jh)^3 ) )

where R is the 2x2 correlation matrix of the two predictors and rbar is their mean
correlation. It is compared against a t distribution with n - 3 degrees of freedom.

Predictions are taken from the same validation-locked selection protocol used
elsewhere in the revision: for each dataset x trait the EGT variant and the
baseline are the seed-42 validation winners, and per-animal predictions are the
mean across the five replicate seeds.

Run from the project root:
    python scripts/dependent_correlation_test.py
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
from paired_bootstrap_ci import EGT_MODELS, BASELINES, SEEDS, load_cell  # noqa: E402

RESULT = OUT / "dependent_correlation_test.csv"
REPORT = OUT / "dependent_correlation_report.txt"


def williams_t(r_jk: float, r_jh: float, r_kh: float, n: int) -> tuple[float, float]:
    """Williams' t for r_jk vs r_jh sharing variable j; returns (t, two-sided p)."""
    det = 1.0 - r_jk**2 - r_jh**2 - r_kh**2 + 2.0 * r_jk * r_jh * r_kh
    rbar = (r_jk + r_jh) / 2.0
    denom = 2.0 * ((n - 1) / (n - 3)) * det + (rbar**2) * (1.0 - r_kh) ** 3
    if denom <= 0:
        return float("nan"), float("nan")
    t = (r_jk - r_jh) * math.sqrt((n - 1) * (1.0 + r_kh) / denom)
    # two-sided p from Student's t with n-3 df
    try:
        from scipy import stats
        p = float(2.0 * stats.t.sf(abs(t), df=n - 3))
    except Exception:
        p = float("nan")
    return float(t), p


def main() -> None:
    lock = json.loads((OUT / "selection_lock.json").read_text(encoding="utf-8"))
    locked = {(w["dataset"], w["trait"], w["family"]): w["model"] for w in lock["winners"]}

    rows = []
    for dataset, traits in TRAITS.items():
        for trait in traits:
            egt = locked.get((dataset, trait, "EGT"))
            base = locked.get((dataset, trait, "baseline"))
            if egt is None or base is None:
                continue
            per_model = load_cell(dataset, trait)
            if egt not in per_model or base not in per_model:
                continue
            if len(per_model[egt]) != len(SEEDS) or len(per_model[base]) != len(SEEDS):
                continue
            y = per_model[egt][0]["y_true"].to_numpy(dtype=np.float64)
            p_egt = np.mean([f["y_pred"].to_numpy(dtype=np.float64) for f in per_model[egt]], axis=0)
            p_base = np.mean([f["y_pred"].to_numpy(dtype=np.float64) for f in per_model[base]], axis=0)
            n = len(y)

            def corr(a, b):
                a, b = a - a.mean(), b - b.mean()
                den = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
                return float((a * b).sum() / den) if den > 0 else float("nan")

            r_y_egt = corr(y, p_egt)
            r_y_base = corr(y, p_base)
            r_egt_base = corr(p_egt, p_base)
            t, p = williams_t(r_y_egt, r_y_base, r_egt_base, n)
            rows.append({
                "dataset": dataset, "trait": trait, "n": n,
                "egt_model": egt, "baseline_model": base,
                "r_true_egt": r_y_egt, "r_true_baseline": r_y_base,
                "r_egt_baseline": r_egt_base,
                "diff": r_y_egt - r_y_base,
                "williams_t": t, "p_value": p,
                "significant_0_05": bool(np.isfinite(p) and p < 0.05),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No comparable cells found.")
    df = df.sort_values(["dataset", "trait"]).reset_index(drop=True)
    df.to_csv(RESULT, index=False)

    lines = ["=" * 100,
             "Williams' test for dependent correlations (Reviewer 2, Comment 3)",
             "=" * 100, "",
             "Validation-locked pair (seed-42 validation winners); per-animal prediction is the",
             "mean over 5 replicate seeds. Correlations share the phenotype, so they are dependent",
             "and the test uses n-3 degrees of freedom.", "",
             f"{'dataset':<11} {'trait':<7} {'n':>4} {'EGT':<7} {'base':<8} {'r_egt':>8} {'r_base':>8} {'diff':>8} {'t':>8} {'p':>9} {'sig':<4}",
             "-" * 100]
    for r in df.itertuples(index=False):
        lines.append(
            f"{r.dataset:<11} {r.trait:<7} {r.n:>4} {r.egt_model:<7} {r.baseline_model:<8} "
            f"{r.r_true_egt:>8.4f} {r.r_true_baseline:>8.4f} {r.diff:>+8.4f} {r.williams_t:>8.3f} "
            f"{r.p_value:>9.4f} {'YES' if r.significant_0_05 else 'no':<4}"
        )
    lines.append("")
    lines.append(f"cells tested: {len(df)}   significant at 0.05: {int(df.significant_0_05.sum())}")
    lines.append("")
    lines.append("Where significant, direction:")
    for r in df[df.significant_0_05].itertuples(index=False):
        who = "EGT higher" if r.diff > 0 else "baseline higher"
        lines.append(f"  {r.dataset} {r.trait}: {who} (diff {r.diff:+.4f}, p={r.p_value:.4f})")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
