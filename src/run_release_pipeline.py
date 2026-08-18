"""Run EGT training, test evaluation, and optional explainability from released inputs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DATASETS = ("PIC", "HZA", "BloodLipid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate EGT from an extracted EGT_model_inputs_v1 archive."
    )
    parser.add_argument("--data-root", type=Path, required=True, help="Directory containing PIC, HZA, and BloodLipid.")
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--mode", choices=("single", "mtl"), default="single")
    parser.add_argument("--trait", type=str, help="Required for single-trait EGT.")
    parser.add_argument("--traits", nargs="+", help="Optional trait list for multi-task EGT.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto", help="Torch device, for example auto, cuda, or cpu.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--ae-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--representation", choices=("autoencoder", "chunk", "fused"), default="autoencoder")
    parser.add_argument("--loss-association-metric", choices=("pearson", "spearman", "r2"), default="pearson")
    parser.add_argument("--explain", action="store_true", help="Run integrated gradients and masking after a single-trait run.")
    parser.add_argument("--ig-steps", type=int, default=24)
    parser.add_argument("--window-size", type=int, default=256)
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def validate_inputs(data_dir: Path) -> None:
    required = [
        data_dir / "genotypes_train.csv",
        data_dir / "genotypes_valid.csv",
        data_dir / "genotypes_test.csv",
        data_dir / "labels_long.csv",
        data_dir / "markers.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Released model-input files are missing:\n" + "\n".join(missing))


def main() -> None:
    args = parse_args()
    if args.mode == "single" and not args.trait:
        raise ValueError("--trait is required when --mode single.")
    if args.explain and args.mode != "single":
        raise ValueError("--explain is available for a single-trait EGT checkpoint only.")

    data_dir = args.data_root / args.dataset
    validate_inputs(data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).resolve().parent
    egt_command = [
        sys.executable,
        str(script_dir / "egt.py"),
        "--mode",
        args.mode,
        "--genotype-dir",
        str(data_dir),
        "--labels",
        str(data_dir / "labels_long.csv"),
        "--dataset",
        args.dataset,
        "--output-dir",
        str(args.output_dir),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--ae-epochs",
        str(args.ae_epochs),
        "--patience",
        str(args.patience),
        "--alpha",
        str(args.alpha),
        "--representation",
        args.representation,
        "--loss-association-metric",
        args.loss_association_metric,
    ]
    if args.mode == "single":
        egt_command.extend(["--trait", args.trait])
    elif args.traits:
        egt_command.extend(["--traits", *args.traits])

    run(egt_command)

    manifest = {
        "dataset": args.dataset,
        "mode": args.mode,
        "trait": args.trait,
        "traits": args.traits,
        "data_dir": str(data_dir),
        "test_metrics": str(args.output_dir / "summary_metrics.csv"),
        "checkpoint": str(args.output_dir / "best_checkpoint.pt"),
    }

    if args.explain:
        explain_dir = args.output_dir / "explainability"
        explain_command = [
            sys.executable,
            str(script_dir / "explain_egt.py"),
            "--checkpoint",
            str(args.output_dir / "best_checkpoint.pt"),
            "--genotype-dir",
            str(data_dir),
            "--labels",
            str(data_dir / "labels_long.csv"),
            "--trait",
            args.trait,
            "--dataset",
            args.dataset,
            "--device",
            args.device,
            "--ig-steps",
            str(args.ig_steps),
            "--window-size",
            str(args.window_size),
            "--output-dir",
            str(explain_dir),
        ]
        run(explain_command)
        manifest["explainability_dir"] = str(explain_dir)

    (args.output_dir / "release_run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Done. Test metrics: {args.output_dir / 'summary_metrics.csv'}", flush=True)


if __name__ == "__main__":
    main()
