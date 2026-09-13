"""Regenerate the Animals Round 1 summary layer from completed repeat outputs.

Background
----------
The 2026-09-11 queue was interrupted: ``QUEUE_END completed=0 failed=79
total=435`` at 15:31:20, and ``summarize()`` ran immediately afterwards. A retry
(``QUEUE_START total=79`` 17:24:49 -> ``QUEUE_END completed=79 failed=0``
18:27:54) recovered every job, but the derived summary files were never rebuilt.
As a result ``benchmark_status.json`` still reported 356/435 and
``repeat_metrics_summary.csv`` kept a malformed pandas MultiIndex header.

This script rebuilds that derived layer from the per-job ``metrics.json`` and
``cost.json`` files, which are the authoritative per-run records. It:

1. validates that all 435 repeat jobs have complete, parseable output;
2. writes ``repeat_metrics_long.csv`` and a well-formed
   ``repeat_metrics_summary.csv`` (named index columns, no blank header row);
3. writes ``computational_cost_partial.csv``;
4. rewrites ``benchmark_status.json`` and ``failures/STATUS.md``.

Nothing under ``animals_submission_20260826_final/`` is touched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from animals_round1 import OUT, jobs  # noqa: E402

METRICS_KEYS = ["n", "pearson", "rmse", "mae", "r2", "bias", "slope"]
COST_KEYS = [
    "load_seconds",
    "training_seconds_including_ae",
    "inference_seconds",
    "torch_peak_allocated_bytes",
    "peak_memory_scope",
    "parameters",
    "gpu",
    "wall_seconds",
]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def collect() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Read every repeat job output; return (metrics, costs, problems)."""
    repeat_jobs = jobs("repeat")
    records: list[dict] = []
    costs: list[dict] = []
    problems: list[str] = []

    for job in repeat_jobs:
        directory = Path(job["output"])
        for name in ("DONE.json", "metrics.json", "cost.json", "predictions.csv"):
            if not (directory / name).exists():
                problems.append(f"{job['key']}: missing {name}")
        if problems and problems[-1].startswith(job["key"]):
            continue

        try:
            metrics = load_json(directory / "metrics.json")
            cost = load_json(directory / "cost.json")
        except (json.JSONDecodeError, OSError) as exc:  # pragma: no cover
            problems.append(f"{job['key']}: unreadable ({exc})")
            continue

        for m in metrics:
            records.append(dict(dataset=job["dataset"], model=job["model"], seed=job["seed"], **m))
        costs.append(dict(dataset=job["dataset"], model=job["model"], seed=job["seed"], trait=job["trait"], **cost))

    metrics_df = pd.DataFrame(records)
    costs_df = pd.DataFrame(costs)
    return metrics_df, costs_df, problems


def main() -> None:
    metrics_df, costs_df, problems = collect()
    expected = len(jobs("repeat"))

    print("=" * 78)
    print("Animals Round 1 summary regeneration")
    print("=" * 78)
    print(f"expected repeat jobs : {expected}")
    print(f"cost rows collected  : {len(costs_df)}")
    print(f"metric rows collected: {len(metrics_df)}")
    print(f"integrity problems   : {len(problems)}")
    for line in problems[:40]:
        print(f"  PROBLEM: {line}")

    if problems:
        raise SystemExit("Refusing to regenerate summaries while outputs are incomplete.")
    if len(costs_df) != expected:
        raise SystemExit(f"Expected {expected} cost rows, found {len(costs_df)}.")

    # --- long table (unchanged schema, now deterministically ordered) ---
    ordered = ["dataset", "model", "seed", "n", "pearson", "rmse", "mae", "r2", "trait", "split", "bias", "slope"]
    metrics_df = metrics_df.reindex(columns=ordered).sort_values(["dataset", "trait", "model", "split", "seed"]).reset_index(drop=True)
    metrics_df.to_csv(OUT / "repeat_metrics_long.csv", index=False)

    # --- summary table: group keys promoted to real, named columns ---
    grouped = (
        metrics_df.groupby(["dataset", "trait", "model", "split"], dropna=False)[
            ["pearson", "rmse", "mae", "bias", "slope"]
        ]
        .agg(["count", "mean", "std"])
        .round(6)
    )
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    grouped = grouped.reset_index().sort_values(["dataset", "trait", "model", "split"]).reset_index(drop=True)
    grouped.to_csv(OUT / "repeat_metrics_summary.csv", index=False)

    # --- computational cost ---
    cost_ordered = ["dataset", "model", "seed", "trait"] + COST_KEYS
    costs_df = costs_df.reindex(columns=cost_ordered).sort_values(["dataset", "trait", "model", "seed"]).reset_index(drop=True)
    costs_df.to_csv(OUT / "computational_cost_partial.csv", index=False)

    # --- status ---
    complete = len(costs_df) == expected
    (OUT / "benchmark_status.json").write_text(
        json.dumps(
            {
                "expected_jobs": expected,
                "finished_jobs": len(costs_df),
                "complete": complete,
                "regenerated": pd.Timestamp.now().isoformat(timespec="seconds"),
                "source": "per-job metrics.json/cost.json under results/animals_round1/repeat",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    failures_dir = OUT / "failures"
    if failures_dir.exists():
        stale = sorted(p.name for p in failures_dir.glob("*.json"))
        (failures_dir / "STATUS.md").write_text(
            "# Stale failure records - not current status\n\n"
            f"{len(stale)} records in this directory were written by\n"
            "`animals_round1.run_queue` during the interrupted 2026-09-11 run\n"
            "(`QUEUE_END completed=0 failed=79 total=435` at 15:31:20).\n\n"
            "All 79 jobs were subsequently recovered by the retry recorded in\n"
            "`animals_round1_progress.log`:\n\n"
            "```\n[2026-09-11 17:24:49] QUEUE_START total=79 workers=2 gpus=0,1\n"
            "[2026-09-11 18:27:54] QUEUE_END completed=79 failed=0 total=79\n```\n\n"
            "The queue never removes failure records on a later success, so these\n"
            "files are retained only as provenance of the interruption. Current\n"
            "status is in `../benchmark_status.json`.\n",
            encoding="utf-8",
        )

    print()
    print("--- written ---")
    print(f"repeat_metrics_long.csv        rows={len(metrics_df)}")
    print(f"repeat_metrics_summary.csv     rows={len(grouped)}")
    print(f"computational_cost_partial.csv rows={len(costs_df)}")
    print(f"benchmark_status.json          complete={complete}")
    print()
    print("summary columns:", ", ".join(grouped.columns))


if __name__ == "__main__":
    main()
