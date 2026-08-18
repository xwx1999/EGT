from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from egt import (
    EGTConfig,
    EnhancedGenomicTransformer,
    MultiTraitBundle,
    evaluate_predictions,
    get_torch_device,
    load_dataset_bundle,
    predict_arrays,
    read_table,
    single_to_multitrait_bundle,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain a trained BloodLipid fused EGT with Integrated Gradients and masking.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--genotype-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--trait", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="BloodLipid")
    parser.add_argument("--target-column", type=str, default="value")
    parser.add_argument("--id-column", type=str, default="ID")
    parser.add_argument("--trait-column", type=str, default="trait")
    parser.add_argument("--split-column", type=str, default="split")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--ig-steps", type=int, default=24)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--top-fracs", nargs="+", type=float, default=[0.01, 0.05, 0.10])
    parser.add_argument("--mask-random-seed", type=int, default=42)
    parser.add_argument("--random-repeats", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_checkpoint(path: Path):
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def config_from_checkpoint(payload: dict, cli_args: argparse.Namespace) -> EGTConfig:
    config_dict = dict(payload["config"])
    config_dict["device"] = cli_args.device
    config_dict["num_workers"] = cli_args.num_workers
    config_dict["batch_size"] = cli_args.batch_size
    config_dict["amp"] = False
    return EGTConfig(**config_dict)


def load_single_trait_bundle(args: argparse.Namespace) -> MultiTraitBundle:
    bundle = load_dataset_bundle(
        genotype_train=args.genotype_dir / "genotypes_train.csv",
        genotype_valid=args.genotype_dir / "genotypes_valid.csv",
        genotype_test=args.genotype_dir / "genotypes_test.csv",
        genotypes=None,
        labels_path=args.labels,
        id_col=args.id_column,
        target_column=args.target_column,
        trait=args.trait,
        trait_column=args.trait_column,
        split_column=args.split_column,
    )
    return single_to_multitrait_bundle(bundle)


def restore_model(payload: dict, config: EGTConfig, n_markers: int):
    import torch

    model = EnhancedGenomicTransformer(n_markers, 1, config).__call__()
    model.load_state_dict(payload["model_state_dict"])
    device = get_torch_device(config.device)
    model = model.to(device)
    model.eval()
    return model, device


def transform_X(bundle: MultiTraitBundle, checkpoint_payload: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prep = checkpoint_payload["marker_preprocessor"]
    means = np.asarray(prep["marker_means"], dtype=np.float32)
    stds = np.asarray(prep["marker_stds"], dtype=np.float32)

    def tx(X: np.ndarray) -> np.ndarray:
        Xf = np.asarray(X, dtype=np.float32)
        Xf = np.where(np.isnan(Xf), means, Xf)
        if prep["standardize"]:
            Xf = (Xf - means) / stds
        return Xf.astype(np.float32, copy=False)

    return tx(bundle.train.X), tx(bundle.valid.X), tx(bundle.test.X)


def integrated_gradients(
    model,
    inputs: np.ndarray,
    device,
    steps: int,
    batch_size: int,
) -> np.ndarray:
    import torch

    baseline = np.zeros_like(inputs, dtype=np.float32)
    all_attr: list[np.ndarray] = []
    for start in range(0, len(inputs), batch_size):
        x_np = inputs[start : start + batch_size]
        base_np = baseline[start : start + batch_size]
        x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
        b = torch.as_tensor(base_np, dtype=torch.float32, device=device)
        total_grad = torch.zeros_like(x)
        for alpha in torch.linspace(0.0, 1.0, steps, device=device):
            interp = (b + alpha * (x - b)).detach().requires_grad_(True)
            pred = model(interp)[:, 0].sum()
            grad = torch.autograd.grad(pred, interp, retain_graph=False, create_graph=False)[0]
            total_grad = total_grad + grad.detach()
        avg_grad = total_grad / float(steps)
        attr = ((x - b) * avg_grad).detach().cpu().numpy()
        all_attr.append(attr)
    return np.concatenate(all_attr, axis=0).astype(np.float32)


def aggregate_importance(
    marker_names: list[str],
    markers_df: pd.DataFrame,
    attributions: np.ndarray,
    window_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mean_abs = np.mean(np.abs(attributions), axis=0)
    snp_df = pd.DataFrame(
        {
            "marker_id": marker_names,
            "importance": mean_abs,
        }
    )
    if "marker_id" in markers_df.columns:
        snp_df = snp_df.merge(markers_df, on="marker_id", how="left")
    elif len(markers_df) == len(snp_df):
        snp_df = pd.concat([snp_df, markers_df.reset_index(drop=True)], axis=1)

    if "chromosome" in snp_df.columns and "position" in snp_df.columns:
        snp_df["chromosome"] = pd.to_numeric(snp_df["chromosome"], errors="coerce")
        snp_df["position"] = pd.to_numeric(snp_df["position"], errors="coerce")
        snp_df["window_start"] = (snp_df["position"] // window_size) * window_size
        snp_df["window_end"] = snp_df["window_start"] + window_size
        window_df = (
            snp_df.dropna(subset=["chromosome", "position"])
            .groupby(["chromosome", "window_start", "window_end"], as_index=False)
            .agg(
                window_importance=("importance", "sum"),
                n_snps=("marker_id", "count"),
            )
            .sort_values("window_importance", ascending=False)
            .reset_index(drop=True)
        )
    else:
        snp_df["window_index"] = np.arange(len(snp_df)) // window_size
        window_df = (
            snp_df.groupby("window_index", as_index=False)
            .agg(window_importance=("importance", "sum"), n_snps=("marker_id", "count"))
            .sort_values("window_importance", ascending=False)
            .reset_index(drop=True)
        )

    snp_df = snp_df.sort_values("importance", ascending=False).reset_index(drop=True)
    return snp_df, window_df


def evaluate_masking(
    model,
    bundle: MultiTraitBundle,
    X_test_std: np.ndarray,
    config: EGTConfig,
    checkpoint_payload: dict,
    markers_df: pd.DataFrame,
    window_df: pd.DataFrame,
    top_fracs: list[float],
    window_size: int,
    random_seed: int,
    random_repeats: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    device = get_torch_device(config.device)
    y_means = np.asarray(checkpoint_payload["y_means"], dtype=np.float32)
    y_stds = np.asarray(checkpoint_payload["y_stds"], dtype=np.float32)

    rows: list[dict[str, object]] = []
    n_windows = len(window_df)
    base_pred_z = predict_arrays(model, X_test_std, config.batch_size, config.num_workers, device, config.amp)
    base_metrics = evaluate_predictions(bundle, bundle.test, base_pred_z, y_means, y_stds)[bundle.traits[0]]

    if "chromosome" in markers_df.columns and "position" in markers_df.columns and "window_start" in markers_df.columns:
        marker_windows = markers_df[["marker_id", "window_start"]].copy()
        marker_windows["window_key"] = marker_windows["window_start"].astype(str)
        window_df = window_df.copy()
        window_df["window_key"] = window_df["window_start"].astype(str)
    else:
        marker_windows = pd.DataFrame({"marker_id": markers_df["marker_id"], "window_key": markers_df["window_index"].astype(str)})
        window_df = window_df.copy()
        window_df["window_key"] = window_df["window_index"].astype(str)

    marker_to_idx = {marker: idx for idx, marker in enumerate(bundle.marker_names)}
    windows_ranked = window_df["window_key"].tolist()
    for frac in top_fracs:
        k = max(1, int(round(n_windows * frac)))
        top_keys = set(windows_ranked[:k])
        low_keys = set(windows_ranked[-k:])
        strategy_groups: list[tuple[str, set[str], int]] = [("top", top_keys, 0), ("low", low_keys, 0)]
        for repeat_idx in range(max(1, random_repeats)):
            random_keys = set(rng.choice(windows_ranked, size=k, replace=False).tolist())
            strategy_groups.append(("random", random_keys, repeat_idx))
        for strategy, keys, repeat_idx in strategy_groups:
            selected_markers = marker_windows[marker_windows["window_key"].isin(keys)]["marker_id"].tolist()
            selected_idx = [marker_to_idx[m] for m in selected_markers if m in marker_to_idx]
            X_masked = X_test_std.copy()
            if selected_idx:
                X_masked[:, selected_idx] = 0.0
            pred_z = predict_arrays(model, X_masked, config.batch_size, config.num_workers, device, config.amp)
            metrics = evaluate_predictions(bundle, bundle.test, pred_z, y_means, y_stds)[bundle.traits[0]]
            rows.append(
                {
                    "trait": bundle.traits[0],
                    "mask_fraction": frac,
                    "strategy": strategy,
                    "repeat": repeat_idx,
                    "n_windows": k,
                    "n_markers_masked": len(selected_idx),
                    "baseline_PCC": base_metrics["pearson"],
                    "masked_PCC": metrics["pearson"],
                    "delta_PCC": metrics["pearson"] - base_metrics["pearson"],
                    "masked_RMSE": metrics["rmse"],
                    "masked_MAE": metrics["mae"],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_payload = load_checkpoint(args.checkpoint)
    config = config_from_checkpoint(checkpoint_payload, args)
    bundle = load_single_trait_bundle(args)
    model, _device = restore_model(checkpoint_payload, config, len(bundle.marker_names))

    X_train_std, X_valid_std, X_test_std = transform_X(bundle, checkpoint_payload)
    attributions = integrated_gradients(model, X_test_std, get_torch_device(config.device), args.ig_steps, args.batch_size)

    markers_path = args.genotype_dir / "markers.csv"
    markers_df = read_table(markers_path) if markers_path.exists() else pd.DataFrame({"marker_id": bundle.marker_names})
    snp_df, window_df = aggregate_importance(bundle.marker_names, markers_df, attributions, args.window_size)
    mask_df = evaluate_masking(
        model,
        bundle,
        X_test_std,
        config,
        checkpoint_payload,
        snp_df,
        window_df,
        args.top_fracs,
        args.window_size,
        args.mask_random_seed,
        args.random_repeats,
    )

    np.save(args.output_dir / "attributions_test.npy", attributions)
    snp_df.to_csv(args.output_dir / "snp_importance.csv", index=False)
    window_df.to_csv(args.output_dir / "window_importance.csv", index=False)
    mask_df.to_csv(args.output_dir / "masking_validation.csv", index=False)
    write_json(
        {
            "dataset": args.dataset,
            "trait": args.trait,
            "checkpoint": str(args.checkpoint),
            "ig_steps": args.ig_steps,
            "window_size": args.window_size,
            "top_fracs": args.top_fracs,
            "random_repeats": args.random_repeats,
        },
        args.output_dir / "explainability_manifest.json",
    )


if __name__ == "__main__":
    main()
