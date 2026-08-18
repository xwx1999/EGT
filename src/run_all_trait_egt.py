from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


EXPERIMENTS = [
    {
        "dataset": "HZA",
        "genotype_dir": Path("data/processed/HZA"),
        "labels": Path("data/processed/HZA/labels_long.csv"),
        "target_column": "value",
        "single": {"learning_rate": 5e-4, "batch_size": 64, "dropout": 0.10, "epochs": 120},
        "single_traits": {
            "MORT1": {"epochs": 120},
            "MORT2": {"epochs": 150},
            "MORT3": {"epochs": 120},
        },
        "mtl": {"learning_rate": 5e-4, "batch_size": 64, "dropout": 0.10, "epochs": 150},
    },
    {
        "dataset": "PIC",
        "genotype_dir": Path("data/processed/PIC"),
        "labels": Path("data/processed/PIC/labels_long.csv"),
        "target_column": "value",
        "single": {"learning_rate": 1e-3, "batch_size": 256, "dropout": 0.15, "epochs": 140},
        "single_traits": {
            "t1": {"epochs": 140},
            "t2": {"epochs": 140},
            "t3": {"epochs": 120},
            "t4": {"epochs": 120},
            "t5": {"epochs": 170},
        },
        "mtl": {"learning_rate": 1e-3, "batch_size": 256, "dropout": 0.15, "epochs": 170},
    },
    {
        "dataset": "BloodLipid",
        "genotype_dir": Path("data/processed/BloodLipid"),
        "labels": Path("data/processed/BloodLipid/labels_long.csv"),
        "target_column": "value",
        "single": {"learning_rate": 5e-4, "batch_size": 64, "dropout": 0.10, "epochs": 120},
        "mtl": {"learning_rate": 5e-4, "batch_size": 64, "dropout": 0.10, "epochs": 150},
    },
]


def discover_traits(labels_path: Path) -> list[str]:
    labels = pd.read_csv(labels_path, usecols=["trait"])
    return sorted(labels["trait"].dropna().astype(str).unique().tolist())


def safe_path_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return safe or "unnamed"


def append_summary(summary_rows: list[dict[str, object]], summary_path: Path, dataset: str, mode: str) -> None:
    if not summary_path.exists():
        return
    summary = pd.read_csv(summary_path)
    summary = summary[summary["split"] == "test"].copy()
    for _, row in summary.iterrows():
        summary_rows.append(
            {
                "dataset": dataset,
                "trait": row["trait"],
                "model": row["model"],
                "mode": mode,
                "n_test": int(row["n"]),
                "PCC": float(row["PCC"]),
                "RMSE": float(row["RMSE"]),
                "MAE": float(row["MAE"]),
            }
        )


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        path.write_text("", encoding="utf-8")
        return
    formatted = df.copy()
    for column in ["PCC", "RMSE", "MAE"]:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(lambda value: "" if pd.isna(value) else f"{value:.6f}")
    formatted = formatted.astype(str)
    columns = formatted.columns.tolist()
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in formatted.iterrows():
        lines.append("| " + " | ".join(row[column] for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dataset_args(config: dict[str, object]) -> list[str]:
    return [
        "--learning-rate",
        str(config["learning_rate"]),
        "--batch-size",
        str(config["batch_size"]),
        "--dropout",
        str(config["dropout"]),
        "--epochs",
        str(config["epochs"]),
    ]


def merged_trait_config(experiment: dict[str, object], trait: str) -> dict[str, object]:
    config = dict(experiment["single"])
    trait_overrides = experiment.get("single_traits", {})
    if isinstance(trait_overrides, dict):
        config.update(trait_overrides.get(trait, {}))
    return config


def resolve_data_paths(experiment: dict[str, object], data_root: Path) -> dict[str, object]:
    resolved = dict(experiment)
    dataset_dir = data_root / str(experiment["dataset"])
    resolved["genotype_dir"] = dataset_dir
    resolved["labels"] = dataset_dir / "labels_long.csv"
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-trait and MTL EGT on all processed traits.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing the PIC, HZA, and BloodLipid input directories.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("results/egt"))
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--datasets", nargs="+", choices=[item["dataset"] for item in EXPERIMENTS])
    parser.add_argument("--modes", nargs="+", choices=["single", "mtl"], default=["single", "mtl"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ae-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--representation", choices=["autoencoder", "chunk", "fused"], default="autoencoder")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--loss-variant", choices=["hybrid", "corr_only", "mse_only"], default="hybrid")
    parser.add_argument("--n-tokens", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--encoder-hidden-dim", type=int, default=1024)
    parser.add_argument("--head-hidden-dim", type=int, default=128)
    parser.add_argument("--mtl-balance-power", type=float, default=0.5)
    parser.add_argument("--mtl-selection-metric", choices=["mean", "median", "min"], default="mean")
    parser.add_argument("--no-token-projection-bias", action="store_true")
    parser.add_argument("--refit-using-valid", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = set(args.datasets) if args.datasets else None
    args.output_root.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    summary_path = args.output_root / "summary_metrics.csv"
    summary_rows: list[dict[str, object]] = []
    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        selected_datasets = selected if selected is not None else {item["dataset"] for item in EXPERIMENTS}
        replace_mask = existing["dataset"].isin(selected_datasets) & existing["mode"].isin(args.modes)
        summary_rows.extend(existing.loc[~replace_mask].to_dict("records"))

    for experiment in EXPERIMENTS:
        experiment = resolve_data_paths(experiment, args.data_root)
        dataset = experiment["dataset"]
        if selected is not None and dataset not in selected:
            continue
        traits = discover_traits(experiment["labels"])

        if "single" in args.modes:
            for trait in traits:
                run_name = safe_path_name(f"{dataset}_{trait}_single_egt")
                out_dir = args.output_root / "single" / dataset / safe_path_name(trait)
                metrics_path = out_dir / "metrics_all.json"
                if metrics_path.exists() and not args.force:
                    print(f"Skipping {run_name}; existing metrics found.")
                    append_summary(summary_rows, out_dir / "summary_metrics.csv", dataset, "single")
                    continue

                command = [
                    str(args.python),
                    "src/egt.py",
                    "--mode",
                    "single",
                    "--genotype-dir",
                    str(experiment["genotype_dir"]),
                    "--labels",
                    str(experiment["labels"]),
                    "--trait",
                    trait,
                    "--target-column",
                    experiment["target_column"],
                    "--dataset",
                    dataset,
                    "--output-dir",
                    str(out_dir),
                    "--seed",
                    str(args.seed),
                    "--device",
                    args.device,
                    "--ae-epochs",
                    str(args.ae_epochs),
                    "--patience",
                    str(args.patience),
                    "--representation",
                    args.representation,
                    "--chunk-size",
                    str(args.chunk_size),
                    "--loss-variant",
                    args.loss_variant,
                    "--alpha",
                    str(args.alpha),
                    "--n-tokens",
                    str(args.n_tokens),
                    "--d-model",
                    str(args.d_model),
                    "--num-heads",
                    str(args.num_heads),
                    "--num-layers",
                    str(args.num_layers),
                    "--ff-dim",
                    str(args.ff_dim),
                    "--encoder-hidden-dim",
                    str(args.encoder_hidden_dim),
                    "--head-hidden-dim",
                    str(args.head_hidden_dim),
                    "--num-workers",
                    str(args.num_workers),
                    *dataset_args(merged_trait_config(experiment, trait)),
                ]
                if args.no_amp:
                    command.append("--no-amp")
                if args.data_parallel:
                    command.append("--data-parallel")
                if args.no_token_projection_bias:
                    command.append("--no-token-projection-bias")
                if args.refit_using_valid:
                    command.append("--refit-using-valid")

                print(f"Running {run_name}...")
                out_dir.mkdir(parents=True, exist_ok=True)
                with (log_dir / f"{run_name}.out.log").open("w", encoding="utf-8") as stdout, (
                    log_dir / f"{run_name}.err.log"
                ).open("w", encoding="utf-8") as stderr:
                    completed = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
                if completed.returncode != 0:
                    raise RuntimeError(f"{run_name} failed. See {log_dir / f'{run_name}.err.log'}")
                append_summary(summary_rows, out_dir / "summary_metrics.csv", dataset, "single")
                pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

        if "mtl" in args.modes:
            run_name = safe_path_name(f"{dataset}_mtl_egt")
            out_dir = args.output_root / "mtl" / dataset
            metrics_path = out_dir / "metrics_all.json"
            if metrics_path.exists() and not args.force:
                print(f"Skipping {run_name}; existing metrics found.")
                append_summary(summary_rows, out_dir / "summary_metrics.csv", dataset, "mtl")
                continue

            command = [
                str(args.python),
                "src/egt.py",
                "--mode",
                "mtl",
                "--genotype-dir",
                str(experiment["genotype_dir"]),
                "--labels",
                str(experiment["labels"]),
                "--target-column",
                experiment["target_column"],
                "--dataset",
                dataset,
                "--output-dir",
                str(out_dir),
                "--seed",
                str(args.seed),
                "--device",
                args.device,
                "--ae-epochs",
                str(args.ae_epochs),
                "--patience",
                str(args.patience),
                "--representation",
                args.representation,
                "--chunk-size",
                str(args.chunk_size),
                "--loss-variant",
                args.loss_variant,
                "--alpha",
                str(args.alpha),
                "--n-tokens",
                str(args.n_tokens),
                "--d-model",
                str(args.d_model),
                "--num-heads",
                str(args.num_heads),
                "--num-layers",
                str(args.num_layers),
                "--ff-dim",
                str(args.ff_dim),
                "--encoder-hidden-dim",
                str(args.encoder_hidden_dim),
                "--head-hidden-dim",
                str(args.head_hidden_dim),
                "--mtl-balance-power",
                str(args.mtl_balance_power),
                "--mtl-selection-metric",
                args.mtl_selection_metric,
                "--num-workers",
                str(args.num_workers),
                *dataset_args(experiment["mtl"]),
            ]
            if args.no_amp:
                command.append("--no-amp")
            if args.data_parallel:
                command.append("--data-parallel")
            if args.no_token_projection_bias:
                command.append("--no-token-projection-bias")
            if args.refit_using_valid:
                command.append("--refit-using-valid")

            print(f"Running {run_name}...")
            out_dir.mkdir(parents=True, exist_ok=True)
            with (log_dir / f"{run_name}.out.log").open("w", encoding="utf-8") as stdout, (
                log_dir / f"{run_name}.err.log"
            ).open("w", encoding="utf-8") as stderr:
                completed = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
            if completed.returncode != 0:
                raise RuntimeError(f"{run_name} failed. See {log_dir / f'{run_name}.err.log'}")
            append_summary(summary_rows, out_dir / "summary_metrics.csv", dataset, "mtl")
            pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(summary_path, index=False)
    if not summary.empty:
        pivot = summary.pivot_table(index=["dataset", "trait"], columns="model", values="PCC", aggfunc="first")
        pivot.to_csv(args.output_root / "summary_pcc_pivot.csv")
        write_markdown_table(summary, args.output_root / "summary_metrics.md")
    print(f"Done. EGT summary written to {args.output_root / 'summary_metrics.csv'}")


if __name__ == "__main__":
    main()
