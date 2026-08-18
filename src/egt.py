from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from baselines import (
    MarkerPreprocessor,
    SPLIT_LABELS,
    align_split,
    load_dataset_bundle,
    normalize_id,
    pearson_corr,
    read_genotype_matrix,
    read_table,
    regression_metrics,
)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def safe_float(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


@dataclass
class EGTConfig:
    epochs: int = 140
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 25
    dropout: float = 0.15
    alpha: float = 0.15
    n_tokens: int = 8
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 2
    ff_dim: int = 256
    encoder_hidden_dim: int = 1024
    head_hidden_dim: int = 128
    ae_epochs: int = 30
    ae_learning_rate: float = 1e-3
    ae_weight_decay: float = 1e-5
    ae_patience: int = 8
    grad_clip: float = 5.0
    device: str = "auto"
    seed: int = 42
    num_workers: int = 0
    amp: bool = True
    data_parallel: bool = False
    representation: str = "autoencoder"
    chunk_size: int = 0
    mtl_balance_power: float = 0.5
    mtl_selection_metric: str = "mean"
    loss_variant: str = "hybrid"
    loss_association_metric: str = "pearson"
    spearman_temperature: float = 0.1
    token_projection_bias: bool = True
    refit_mode: bool = False


@dataclass
class MultiTraitSplitData:
    ids: np.ndarray
    X: np.ndarray
    y: np.ndarray
    mask: np.ndarray


@dataclass
class MultiTraitBundle:
    train: MultiTraitSplitData
    valid: MultiTraitSplitData | None
    test: MultiTraitSplitData | None
    marker_names: list[str]
    traits: list[str]
    target_column: str


class GenotypeMultiTraitDataset:
    def __init__(self, X: np.ndarray, y: np.ndarray, mask: np.ndarray):
        import torch

        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.mask = torch.as_tensor(mask, dtype=torch.bool)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.mask[idx]


class GenotypeOnlyDataset:
    def __init__(self, X: np.ndarray):
        import torch

        self.X = torch.as_tensor(X, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx]


def make_supervised_loader(
    X: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
):
    from torch.utils.data import DataLoader

    return DataLoader(
        GenotypeMultiTraitDataset(X, y, mask),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def make_genotype_loader(X: np.ndarray, batch_size: int, shuffle: bool, num_workers: int):
    from torch.utils.data import DataLoader

    return DataLoader(
        GenotypeOnlyDataset(X),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def get_torch_device(device: str):
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def autocast_context(device, enabled: bool):
    import torch

    if not enabled or device.type != "cuda":
        return torch.amp.autocast(device_type=device.type, enabled=False)
    return torch.amp.autocast(device_type="cuda", enabled=True)


def load_multitrait_labels(
    path: Path,
    id_col: str,
    target_column: str,
    traits: list[str] | None,
    trait_column: str,
    split_column: str,
) -> tuple[pd.DataFrame, list[str]]:
    labels = read_table(path)
    required = {id_col, trait_column, target_column}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"Label file {path} is missing required columns: {missing}")

    labels = labels.copy()
    labels[id_col] = labels[id_col].map(normalize_id)
    labels[trait_column] = labels[trait_column].astype(str)
    labels[target_column] = pd.to_numeric(labels[target_column], errors="coerce")
    labels = labels[(labels[id_col] != "") & labels[target_column].notna()].copy()

    if traits:
        trait_set = set(map(str, traits))
        labels = labels[labels[trait_column].isin(trait_set)].copy()
        trait_names = list(map(str, traits))
    else:
        trait_names = sorted(labels[trait_column].dropna().astype(str).unique().tolist())

    if labels.empty:
        raise ValueError("No labels remain for the requested MTL traits.")
    if split_column not in labels.columns:
        raise ValueError(f"MTL-EGT requires labels with a {split_column!r} column.")

    labels = labels.groupby([id_col, split_column, trait_column], as_index=False).agg({target_column: "mean"})
    labels = labels.rename(columns={id_col: "ID"})
    return labels, trait_names


def build_multitrait_split(
    ids: np.ndarray,
    X: np.ndarray,
    labels: pd.DataFrame,
    traits: list[str],
    target_column: str,
    split_name: str,
    split_column: str,
    trait_column: str,
) -> MultiTraitSplitData | None:
    label_df = labels[labels[split_column] == split_name].copy()
    if label_df.empty:
        return None

    trait_to_idx = {trait: idx for idx, trait in enumerate(traits)}
    id_to_row = {sample_id: idx for idx, sample_id in enumerate(ids)}
    grouped: dict[str, list[tuple[int, float]]] = {}
    for sample_id_raw, trait_raw, target_value in label_df[["ID", trait_column, target_column]].to_numpy():
        sample_id = normalize_id(sample_id_raw)
        trait = str(trait_raw)
        if sample_id not in id_to_row or trait not in trait_to_idx:
            continue
        grouped.setdefault(sample_id, []).append((trait_to_idx[trait], float(target_value)))

    if not grouped:
        return None

    keep_ids = [sample_id for sample_id in ids if sample_id in grouped]
    row_indices = [id_to_row[sample_id] for sample_id in keep_ids]
    y = np.full((len(keep_ids), len(traits)), np.nan, dtype=np.float32)
    for row_idx, sample_id in enumerate(keep_ids):
        for trait_idx, target_value in grouped[sample_id]:
            y[row_idx, trait_idx] = target_value

    mask = np.isfinite(y)
    y_filled = np.where(mask, y, 0.0).astype(np.float32)
    return MultiTraitSplitData(
        ids=np.asarray(keep_ids, dtype=str),
        X=X[np.asarray(row_indices, dtype=int)],
        y=y_filled,
        mask=mask.astype(bool),
    )


def load_multitrait_bundle(
    genotype_train: Path,
    genotype_valid: Path | None,
    genotype_test: Path | None,
    labels_path: Path,
    id_col: str,
    target_column: str,
    traits: list[str] | None,
    trait_column: str,
    split_column: str,
) -> MultiTraitBundle:
    labels, trait_names = load_multitrait_labels(labels_path, id_col, target_column, traits, trait_column, split_column)

    train_ids, X_train, marker_names = read_genotype_matrix(genotype_train, id_col)
    train = build_multitrait_split(train_ids, X_train, labels, trait_names, target_column, "train", split_column, trait_column)
    if train is None:
        raise ValueError("No training labels overlap MTL genotype IDs.")

    valid = None
    if genotype_valid is not None:
        valid_ids, X_valid, valid_markers = read_genotype_matrix(genotype_valid, id_col)
        if valid_markers != marker_names:
            raise ValueError("Validation genotype markers do not match training markers.")
        valid = build_multitrait_split(valid_ids, X_valid, labels, trait_names, target_column, "valid", split_column, trait_column)

    test = None
    if genotype_test is not None:
        test_ids, X_test, test_markers = read_genotype_matrix(genotype_test, id_col)
        if test_markers != marker_names:
            raise ValueError("Test genotype markers do not match training markers.")
        test = build_multitrait_split(test_ids, X_test, labels, trait_names, target_column, "test", split_column, trait_column)

    return MultiTraitBundle(
        train=train,
        valid=valid,
        test=test,
        marker_names=marker_names,
        traits=trait_names,
        target_column=target_column,
    )


def merge_multitrait_train_valid(bundle: MultiTraitBundle) -> MultiTraitBundle:
    if bundle.valid is None:
        return bundle

    train = bundle.train
    valid = bundle.valid
    merged_train = MultiTraitSplitData(
        ids=np.concatenate([train.ids, valid.ids], axis=0),
        X=np.concatenate([train.X, valid.X], axis=0),
        y=np.concatenate([train.y, valid.y], axis=0),
        mask=np.concatenate([train.mask, valid.mask], axis=0),
    )
    return MultiTraitBundle(
        train=merged_train,
        valid=None,
        test=bundle.test,
        marker_names=bundle.marker_names,
        traits=bundle.traits,
        target_column=bundle.target_column,
    )


def single_to_multitrait_bundle(bundle) -> MultiTraitBundle:
    def wrap(split):
        if split is None:
            return None
        y = split.y.reshape(-1, 1).astype(np.float32)
        return MultiTraitSplitData(ids=split.ids, X=split.X, y=y, mask=np.isfinite(y))

    return MultiTraitBundle(
        train=wrap(bundle.train),
        valid=wrap(bundle.valid),
        test=wrap(bundle.test),
        marker_names=bundle.marker_names,
        traits=[bundle.trait_name],
        target_column=bundle.target_column,
    )


def standardize_targets(train_y: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = np.zeros(train_y.shape[1], dtype=np.float32)
    stds = np.ones(train_y.shape[1], dtype=np.float32)
    for idx in range(train_y.shape[1]):
        values = train_y[train_mask[:, idx], idx]
        if len(values) > 0:
            means[idx] = float(np.mean(values))
            std = float(np.std(values))
            stds[idx] = std if std > 1e-8 else 1.0
    return means, stds


def apply_target_standardization(split: MultiTraitSplitData, y_means: np.ndarray, y_stds: np.ndarray) -> np.ndarray:
    y_z = np.zeros_like(split.y, dtype=np.float32)
    y_z[split.mask] = ((split.y - y_means) / y_stds)[split.mask]
    return y_z


class GenotypeEncoder:
    def __init__(self, n_markers: int, latent_dim: int, hidden_dim: int, dropout: float):
        import torch.nn as nn

        class _Encoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(n_markers, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, latent_dim),
                    nn.LayerNorm(latent_dim),
                )

            def forward(self, x):
                return self.net(x)

        self.model = _Encoder()

    def __call__(self):
        return self.model


class ChunkedGenotypeTokenizer:
    def __init__(self, n_markers: int, n_tokens: int, chunk_size: int, d_model: int, dropout: float, use_bias: bool):
        import torch.nn as nn

        class _Tokenizer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.n_markers = n_markers
                if chunk_size > 0:
                    self.chunk_size = int(chunk_size)
                    self.n_tokens = int(math.ceil(n_markers / self.chunk_size))
                else:
                    self.n_tokens = n_tokens
                    self.chunk_size = int(math.ceil(n_markers / n_tokens))
                self.padded_markers = self.chunk_size * self.n_tokens
                self.projection = nn.Linear(self.chunk_size, d_model, bias=use_bias)
                self.norm = nn.LayerNorm(d_model)
                self.dropout = nn.Dropout(dropout)

            def forward(self, x):
                import torch.nn.functional as F

                if self.padded_markers > self.n_markers:
                    x = F.pad(x, (0, self.padded_markers - self.n_markers))
                x = x.view(x.shape[0], self.n_tokens, self.chunk_size)
                x = self.projection(x)
                x = self.norm(x)
                return self.dropout(x)

        self.model = _Tokenizer()

    def __call__(self):
        return self.model


class GenotypeAutoencoder:
    def __init__(self, n_markers: int, latent_dim: int, hidden_dim: int, dropout: float):
        import torch.nn as nn

        class _Autoencoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = GenotypeEncoder(n_markers, latent_dim, hidden_dim, dropout).__call__()
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, n_markers),
                )

            def forward(self, x):
                z = self.encoder(x)
                return self.decoder(z)

        self.model = _Autoencoder()

    def __call__(self):
        return self.model


class EnhancedGenomicTransformer:
    def __init__(self, n_markers: int, n_traits: int, config: EGTConfig):
        import torch
        import torch.nn as nn

        latent_dim = config.n_tokens * config.d_model

        class _EGT(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                if config.representation == "chunk" and config.chunk_size > 0:
                    self.n_tokens = int(math.ceil(n_markers / config.chunk_size))
                else:
                    self.n_tokens = config.n_tokens
                self.d_model = config.d_model
                self.representation = config.representation
                if self.representation == "autoencoder":
                    self.encoder = GenotypeEncoder(n_markers, latent_dim, config.encoder_hidden_dim, config.dropout).__call__()
                    self.tokenizer = None
                    self.fusion = None
                elif self.representation == "chunk":
                    self.encoder = None
                    self.tokenizer = ChunkedGenotypeTokenizer(
                        n_markers,
                        self.n_tokens,
                        config.chunk_size,
                        config.d_model,
                        config.dropout,
                        config.token_projection_bias,
                    ).__call__()
                    self.fusion = None
                elif self.representation == "fused":
                    self.encoder = GenotypeEncoder(n_markers, latent_dim, config.encoder_hidden_dim, config.dropout).__call__()
                    # Keep the chunk stream aligned to the same token count so the
                    # paper-facing architecture stays one unified EGT variant.
                    self.tokenizer = ChunkedGenotypeTokenizer(
                        n_markers,
                        self.n_tokens,
                        0,
                        config.d_model,
                        config.dropout,
                        config.token_projection_bias,
                    ).__call__()
                    self.fusion = nn.Sequential(
                        nn.Linear(config.d_model * 2, config.d_model),
                        nn.LayerNorm(config.d_model),
                        nn.GELU(),
                        nn.Dropout(config.dropout),
                    )
                else:
                    raise ValueError(f"Unsupported representation: {self.representation}")
                self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
                self.position_embedding = nn.Parameter(torch.zeros(1, self.n_tokens + 1, config.d_model))
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=config.d_model,
                    nhead=config.num_heads,
                    dim_feedforward=config.ff_dim,
                    dropout=config.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
                self.norm = nn.LayerNorm(config.d_model)
                self.heads = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.LayerNorm(config.d_model),
                            nn.Linear(config.d_model, config.head_hidden_dim),
                            nn.GELU(),
                            nn.Dropout(config.dropout),
                            nn.Linear(config.head_hidden_dim, 1),
                        )
                        for _ in range(n_traits)
                    ]
                )
                nn.init.trunc_normal_(self.cls_token, std=0.02)
                nn.init.trunc_normal_(self.position_embedding, std=0.02)

            def forward(self, x):
                if self.representation == "autoencoder":
                    latent = self.encoder(x)
                    tokens = latent.view(latent.shape[0], self.n_tokens, self.d_model)
                elif self.representation == "fused":
                    latent = self.encoder(x)
                    ae_tokens = latent.view(latent.shape[0], self.n_tokens, self.d_model)
                    chunk_tokens = self.tokenizer(x)
                    fused_tokens = self.fusion(torch.cat([ae_tokens, chunk_tokens], dim=-1))
                    tokens = fused_tokens + 0.5 * (ae_tokens + chunk_tokens)
                else:
                    tokens = self.tokenizer(x)
                cls = self.cls_token.expand(tokens.shape[0], -1, -1)
                tokens = torch.cat([cls, tokens], dim=1)
                tokens = tokens + self.position_embedding
                encoded = self.transformer(tokens)
                pooled = self.norm(encoded[:, 0])
                return torch.cat([head(pooled) for head in self.heads], dim=1)

        self.model = _EGT()

    def __call__(self):
        return self.model


def copy_pretrained_encoder(egt_model, autoencoder) -> None:
    target = egt_model.module if hasattr(egt_model, "module") else egt_model
    source = autoencoder.module if hasattr(autoencoder, "module") else autoencoder
    if getattr(target, "encoder", None) is not None:
        target.encoder.load_state_dict(source.encoder.state_dict())


def differentiable_corr(x, y):
    import torch

    x_centered = x - torch.mean(x)
    y_centered = y - torch.mean(y)
    denom = torch.sqrt(torch.sum(x_centered**2) * torch.sum(y_centered**2) + 1e-8)
    if torch.isfinite(denom) and denom > 0:
        return torch.sum(x_centered * y_centered) / denom
    return x.sum() * 0.0


def soft_rank(values, temperature: float):
    import torch

    temperature = max(float(temperature), 1e-4)
    pairwise = values.unsqueeze(0) - values.unsqueeze(1)
    return 1.0 + torch.sum(torch.sigmoid(pairwise / temperature), dim=1)


def association_score(pred, target, metric: str, spearman_temperature: float):
    import torch

    if metric == "pearson":
        return differentiable_corr(pred, target)
    if metric == "spearman":
        pred_rank = soft_rank(pred, spearman_temperature)
        target_rank = soft_rank(target.detach(), spearman_temperature)
        return differentiable_corr(pred_rank, target_rank)
    if metric == "r2":
        residual = torch.sum((pred - target) ** 2)
        target_var = torch.sum((target - torch.mean(target)) ** 2) + 1e-8
        return 1.0 - residual / target_var
    raise ValueError(f"Unsupported association metric: {metric}")


def masked_hybrid_loss(
    pred,
    target,
    mask,
    alpha: float,
    trait_weights: np.ndarray | None = None,
    variant: str = "hybrid",
    association_metric: str = "pearson",
    spearman_temperature: float = 0.1,
):
    import torch

    mask = mask.bool()
    if not torch.any(mask):
        return pred.sum() * 0.0

    if trait_weights is None:
        weights = torch.ones(pred.shape[1], device=pred.device, dtype=pred.dtype)
    else:
        weights = torch.as_tensor(trait_weights, device=pred.device, dtype=pred.dtype)

    mse_terms = []
    association_terms = []
    for trait_idx in range(pred.shape[1]):
        trait_mask = mask[:, trait_idx]
        if torch.count_nonzero(trait_mask) < 2:
            continue
        weight = weights[trait_idx]
        y = target[trait_mask, trait_idx]
        yhat = pred[trait_mask, trait_idx]
        mse_terms.append(weight * torch.mean((yhat - y) ** 2))
        association_terms.append(weight * association_score(yhat, y, association_metric, spearman_temperature))

    mse = torch.mean(torch.stack(mse_terms)) if mse_terms else pred.sum() * 0.0
    mean_association = torch.mean(torch.stack(association_terms)) if association_terms else pred.sum() * 0.0
    if variant == "corr_only":
        return -mean_association
    if variant == "mse_only":
        return mse
    return alpha * mse - (1.0 - alpha) * mean_association


def predict_arrays(model, X: np.ndarray, batch_size: int, num_workers: int, device, amp: bool) -> np.ndarray:
    import torch

    loader = make_genotype_loader(X, batch_size, shuffle=False, num_workers=num_workers)
    preds: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device, non_blocking=True)
            with autocast_context(device, amp):
                pred = model(xb)
            preds.append(pred.detach().float().cpu().numpy())
    return np.concatenate(preds, axis=0).astype(np.float32)


def evaluate_predictions(
    bundle: MultiTraitBundle,
    split: MultiTraitSplitData,
    pred_z: np.ndarray,
    y_means: np.ndarray,
    y_stds: np.ndarray,
) -> dict[str, dict[str, float]]:
    pred = pred_z * y_stds + y_means
    metrics: dict[str, dict[str, float]] = {}
    for trait_idx, trait in enumerate(bundle.traits):
        mask = split.mask[:, trait_idx]
        if np.count_nonzero(mask) == 0:
            continue
        metrics[trait] = regression_metrics(split.y[mask, trait_idx], pred[mask, trait_idx])
    return metrics


def score_validation(metrics_by_trait: dict[str, dict[str, float]], traits: list[str], mode: str = "mean") -> float:
    scores = [metrics_by_trait[trait]["pearson"] for trait in traits if trait in metrics_by_trait]
    finite = [score for score in scores if np.isfinite(score)]
    if finite:
        if mode == "min":
            return float(np.min(finite))
        if mode == "median":
            return float(np.median(finite))
        return float(np.mean(finite))
    rmses = [metrics_by_trait[trait]["rmse"] for trait in traits if trait in metrics_by_trait]
    finite_rmse = [rmse for rmse in rmses if np.isfinite(rmse)]
    return -float(np.mean(finite_rmse)) if finite_rmse else -np.inf


def pretrain_autoencoder(
    X_train: np.ndarray,
    X_valid: np.ndarray | None,
    config: EGTConfig,
    device,
):
    import torch
    import torch.nn as nn

    latent_dim = config.n_tokens * config.d_model
    model = GenotypeAutoencoder(X_train.shape[1], latent_dim, config.encoder_hidden_dim, config.dropout).__call__().to(device)
    if config.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.ae_learning_rate, weight_decay=config.ae_weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(config.ae_epochs, 1))
    criterion = nn.MSELoss()
    train_loader = make_genotype_loader(X_train, config.batch_size, shuffle=True, num_workers=config.num_workers)
    valid_loader = make_genotype_loader(X_valid, config.batch_size, shuffle=False, num_workers=config.num_workers) if X_valid is not None else None
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")

    best_state = None
    best_loss = float("inf")
    bad_epochs = 0
    history = []

    for epoch in range(1, config.ae_epochs + 1):
        model.train()
        train_losses = []
        for xb in train_loader:
            xb = xb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, config.amp):
                recon = model(xb)
                loss = criterion(recon, xb)
            scaler.scale(loss).backward()
            if config.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach().cpu()))
        scheduler.step()

        if valid_loader is not None:
            model.eval()
            valid_losses = []
            with torch.no_grad():
                for xb in valid_loader:
                    xb = xb.to(device, non_blocking=True)
                    with autocast_context(device, config.amp):
                        valid_losses.append(float(criterion(model(xb), xb).detach().cpu()))
            monitor_loss = float(np.mean(valid_losses))
        else:
            monitor_loss = float(np.mean(train_losses))

        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "monitor_loss": monitor_loss})
        if monitor_loss + 1e-6 < best_loss:
            best_loss = monitor_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= config.ae_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def train_egt(bundle: MultiTraitBundle, config: EGTConfig) -> tuple[object, dict]:
    import torch
    import torch.nn as nn

    set_random_seed(config.seed)
    device = get_torch_device(config.device)

    marker_preprocessor = MarkerPreprocessor(standardize=True)
    X_train = marker_preprocessor.fit_transform(bundle.train.X)
    X_valid = marker_preprocessor.transform(bundle.valid.X) if bundle.valid is not None else None
    X_test = marker_preprocessor.transform(bundle.test.X) if bundle.test is not None else None

    y_means, y_stds = standardize_targets(bundle.train.y, bundle.train.mask)
    y_train_z = apply_target_standardization(bundle.train, y_means, y_stds)
    y_valid_z = apply_target_standardization(bundle.valid, y_means, y_stds) if bundle.valid is not None else None

    trait_counts = bundle.train.mask.sum(axis=0).astype(np.float32)
    trait_weights = np.ones_like(trait_counts, dtype=np.float32)
    finite_counts = np.maximum(trait_counts, 1.0)
    if len(finite_counts) > 1:
        trait_weights = finite_counts ** (-config.mtl_balance_power)
        trait_weights = trait_weights / trait_weights.mean()

    ae_history = []
    ae_model = None
    if config.representation in {"autoencoder", "fused"} and config.ae_epochs > 0:
        ae_model, ae_history = pretrain_autoencoder(X_train, X_valid, config, device)

    model = EnhancedGenomicTransformer(X_train.shape[1], len(bundle.traits), config).__call__().to(device)
    if config.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    if ae_model is not None:
        copy_pretrained_encoder(model, ae_model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(config.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")

    train_loader = make_supervised_loader(
        X_train,
        y_train_z,
        bundle.train.mask,
        config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    best_state = None
    best_score = -np.inf
    bad_epochs = 0
    history = []
    start = time.perf_counter()
    use_validation = bundle.valid is not None and X_valid is not None and not config.refit_mode

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb, maskb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            maskb = maskb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, config.amp):
                pred = model(xb)
                loss = masked_hybrid_loss(
                    pred,
                    yb,
                    maskb,
                    config.alpha,
                    trait_weights=trait_weights if len(bundle.traits) > 1 else None,
                    variant=config.loss_variant,
                    association_metric=config.loss_association_metric,
                    spearman_temperature=config.spearman_temperature,
                )
            scaler.scale(loss).backward()
            if config.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach().cpu()))
        scheduler.step()

        if use_validation:
            valid_pred_z = predict_arrays(model, X_valid, config.batch_size, config.num_workers, device, config.amp)
            valid_metrics = evaluate_predictions(bundle, bundle.valid, valid_pred_z, y_means, y_stds)
            monitor_score = score_validation(valid_metrics, bundle.traits, mode=config.mtl_selection_metric)
        else:
            train_pred_z = predict_arrays(model, X_train, config.batch_size, config.num_workers, device, config.amp)
            valid_metrics = evaluate_predictions(bundle, bundle.train, train_pred_z, y_means, y_stds)
            monitor_score = score_validation(valid_metrics, bundle.traits, mode=config.mtl_selection_metric)

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "monitor_score": monitor_score,
            "valid_pearson_mean": monitor_score,
        }
        history.append(row)
        print(
            f"epoch={epoch} train_loss={row['train_loss']:.6f} "
            f"valid_mean_pcc={monitor_score:.6f}",
            flush=True,
        )

        if use_validation:
            if monitor_score > best_score + 1e-6:
                best_score = monitor_score
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= config.patience:
                    break

    if use_validation and best_state is not None:
        model.load_state_dict(best_state)
    if not use_validation:
        best_score = history[-1]["monitor_score"] if history else float("nan")

    fit_state = {
        "marker_preprocessor": marker_preprocessor,
        "y_means": y_means,
        "y_stds": y_stds,
        "X_train": X_train,
        "X_valid": X_valid,
        "X_test": X_test,
        "history": history,
        "ae_history": ae_history,
        "best_valid_score": best_score,
        "training_seconds": float(time.perf_counter() - start),
        "device": str(device),
        "trait_counts": trait_counts.tolist(),
        "trait_weights": trait_weights.tolist(),
    }
    return model, fit_state


def evaluate_trained_model(model, bundle: MultiTraitBundle, fit_state: dict, config: EGTConfig) -> dict:
    device = get_torch_device(config.device)
    result = {
        "model": "mtl_egt" if len(bundle.traits) > 1 else "egt",
        "traits": bundle.traits,
        "target_column": bundle.target_column,
        "n_markers": len(bundle.marker_names),
        "config": asdict(config),
        "best_valid_score": safe_float(fit_state["best_valid_score"]),
        "training_seconds": fit_state["training_seconds"],
        "device": fit_state["device"],
        "metrics": {},
    }

    for split_name in SPLIT_LABELS:
        split = getattr(bundle, split_name)
        X_split = fit_state.get(f"X_{split_name}")
        if split is None or X_split is None:
            continue
        pred_z = predict_arrays(model, X_split, config.batch_size, config.num_workers, device, config.amp)
        result["metrics"][split_name] = evaluate_predictions(bundle, split, pred_z, fit_state["y_means"], fit_state["y_stds"])
    return result


def save_predictions(model, bundle: MultiTraitBundle, fit_state: dict, config: EGTConfig, output_dir: Path) -> None:
    device = get_torch_device(config.device)
    rows = []
    model_name = "mtl_egt" if len(bundle.traits) > 1 else "egt"
    for split_name in SPLIT_LABELS:
        split = getattr(bundle, split_name)
        X_split = fit_state.get(f"X_{split_name}")
        if split is None or X_split is None:
            continue
        pred_z = predict_arrays(model, X_split, config.batch_size, config.num_workers, device, config.amp)
        pred = pred_z * fit_state["y_stds"] + fit_state["y_means"]
        for trait_idx, trait in enumerate(bundle.traits):
            mask = split.mask[:, trait_idx]
            if np.count_nonzero(mask) == 0:
                continue
            rows.append(
                pd.DataFrame(
                    {
                        "ID": split.ids[mask],
                        "split": split_name,
                        "trait": trait,
                        "y_true": split.y[mask, trait_idx],
                        "y_pred": pred[mask, trait_idx],
                        "model": model_name,
                    }
                )
            )
    if rows:
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.concat(rows, ignore_index=True).to_csv(output_dir / f"predictions_{model_name}.csv", index=False)


def write_training_history(fit_state: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fit_state["history"]).to_csv(output_dir / "training_history.csv", index=False)
    if fit_state["ae_history"]:
        pd.DataFrame(fit_state["ae_history"]).to_csv(output_dir / "autoencoder_history.csv", index=False)


def save_checkpoint(
    model,
    bundle: MultiTraitBundle,
    fit_state: dict,
    config: EGTConfig,
    output_dir: Path,
) -> None:
    import torch

    model_to_save = model.module if hasattr(model, "module") else model
    payload = {
        "model_state_dict": model_to_save.state_dict(),
        "config": asdict(config),
        "marker_names": bundle.marker_names,
        "traits": bundle.traits,
        "target_column": bundle.target_column,
        "marker_preprocessor": {
            "standardize": bool(fit_state["marker_preprocessor"].standardize),
            "marker_means": fit_state["marker_preprocessor"].marker_means_,
            "marker_stds": fit_state["marker_preprocessor"].marker_stds_,
        },
        "y_means": fit_state["y_means"],
        "y_stds": fit_state["y_stds"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_dir / "best_checkpoint.pt")


def flatten_metrics(result: dict, dataset: str | None = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split_name, metrics_by_trait in result["metrics"].items():
        for trait, metrics in metrics_by_trait.items():
            row = {
                "dataset": dataset,
                "trait": trait,
                "model": result["model"],
                "split": split_name,
                "n": metrics["n"],
                "PCC": metrics["pearson"],
                "RMSE": metrics["rmse"],
                "MAE": metrics["mae"],
            }
            rows.append(row)
    return rows


def infer_traits(labels_path: Path, trait_column: str) -> list[str]:
    labels = read_table(labels_path)
    if trait_column not in labels.columns:
        raise ValueError(f"Cannot infer traits because {trait_column!r} is missing from {labels_path}.")
    return sorted(labels[trait_column].dropna().astype(str).unique().tolist())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train single-trait EGT or parallel MTL-EGT.")
    parser.add_argument("--mode", choices=["single", "mtl"], default="single")
    genotype_group = parser.add_mutually_exclusive_group(required=True)
    genotype_group.add_argument("--genotype-dir", type=Path)
    genotype_group.add_argument("--genotype-train", type=Path)
    parser.add_argument("--genotype-valid", type=Path)
    parser.add_argument("--genotype-test", type=Path)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--trait", type=str, help="Trait for single-trait EGT.")
    parser.add_argument("--traits", nargs="+", help="Traits for MTL-EGT. Defaults to all traits in the label file.")
    parser.add_argument("--trait-column", type=str, default="trait")
    parser.add_argument("--target-column", type=str, default="value")
    parser.add_argument("--id-column", type=str, default="ID")
    parser.add_argument("--split-column", type=str, default="split")
    parser.add_argument("--dataset", type=str, help="Dataset name to write into flattened metric rows.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/egt"))

    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--n-tokens", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--encoder-hidden-dim", type=int, default=1024)
    parser.add_argument("--head-hidden-dim", type=int, default=128)
    parser.add_argument("--ae-epochs", type=int, default=30)
    parser.add_argument("--ae-learning-rate", type=float, default=1e-3)
    parser.add_argument("--ae-weight-decay", type=float, default=1e-5)
    parser.add_argument("--ae-patience", type=int, default=8)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--representation", choices=["autoencoder", "chunk", "fused"], default="autoencoder")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--mtl-balance-power", type=float, default=0.5)
    parser.add_argument("--mtl-selection-metric", choices=["mean", "median", "min"], default="mean")
    parser.add_argument("--loss-variant", choices=["hybrid", "corr_only", "mse_only"], default="hybrid")
    parser.add_argument("--loss-association-metric", choices=["pearson", "spearman", "r2"], default="pearson")
    parser.add_argument("--spearman-temperature", type=float, default=0.1)
    parser.add_argument("--no-token-projection-bias", action="store_true")
    parser.add_argument("--refit-using-valid", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> EGTConfig:
    return EGTConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        dropout=args.dropout,
        alpha=args.alpha,
        n_tokens=args.n_tokens,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        encoder_hidden_dim=args.encoder_hidden_dim,
        head_hidden_dim=args.head_hidden_dim,
        ae_epochs=args.ae_epochs,
        ae_learning_rate=args.ae_learning_rate,
        ae_weight_decay=args.ae_weight_decay,
        ae_patience=args.ae_patience,
        grad_clip=args.grad_clip,
        device=args.device,
        seed=args.seed,
        num_workers=args.num_workers,
        amp=not args.no_amp,
        data_parallel=args.data_parallel,
        representation=args.representation,
        chunk_size=args.chunk_size,
        mtl_balance_power=args.mtl_balance_power,
        mtl_selection_metric=args.mtl_selection_metric,
        loss_variant=args.loss_variant,
        loss_association_metric=args.loss_association_metric,
        spearman_temperature=args.spearman_temperature,
        token_projection_bias=not args.no_token_projection_bias,
        refit_mode=args.refit_using_valid,
    )


def resolve_genotype_paths(args: argparse.Namespace) -> tuple[Path, Path | None, Path | None]:
    if args.genotype_dir is not None:
        return (
            args.genotype_dir / "genotypes_train.csv",
            args.genotype_dir / "genotypes_valid.csv",
            args.genotype_dir / "genotypes_test.csv",
        )
    return args.genotype_train, args.genotype_valid, args.genotype_test


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)
    genotype_train, genotype_valid, genotype_test = resolve_genotype_paths(args)
    config = config_from_args(args)

    if args.mode == "single":
        if not args.trait:
            raise ValueError("--trait is required in --mode single.")
        single_bundle = load_dataset_bundle(
            genotype_train=genotype_train,
            genotype_valid=genotype_valid,
            genotype_test=genotype_test,
            genotypes=None,
            labels_path=args.labels,
            id_col=args.id_column,
            target_column=args.target_column,
            trait=args.trait,
            trait_column=args.trait_column,
            split_column=args.split_column,
        )
        bundle = single_to_multitrait_bundle(single_bundle)
    else:
        traits = args.traits if args.traits else infer_traits(args.labels, args.trait_column)
        bundle = load_multitrait_bundle(
            genotype_train=genotype_train,
            genotype_valid=genotype_valid,
            genotype_test=genotype_test,
            labels_path=args.labels,
            id_col=args.id_column,
            target_column=args.target_column,
            traits=traits,
            trait_column=args.trait_column,
            split_column=args.split_column,
        )

    if args.refit_using_valid:
        bundle = merge_multitrait_train_valid(bundle)

    print(
        f"Training {args.mode} EGT on traits={bundle.traits} "
        f"samples(train/valid/test)="
        f"{len(bundle.train.ids)}/"
        f"{len(bundle.valid.ids) if bundle.valid is not None else 0}/"
        f"{len(bundle.test.ids) if bundle.test is not None else 0} "
        f"markers={len(bundle.marker_names)}",
        flush=True,
    )
    model, fit_state = train_egt(bundle, config)
    result = evaluate_trained_model(model, bundle, fit_state, config)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(result, args.output_dir / "metrics_all.json")
    pd.DataFrame(flatten_metrics(result, args.dataset)).to_csv(args.output_dir / "summary_metrics.csv", index=False)
    save_predictions(model, bundle, fit_state, config, args.output_dir)
    write_training_history(fit_state, args.output_dir)
    save_checkpoint(model, bundle, fit_state, config, args.output_dir)

    test_metrics = result["metrics"].get("test", {})
    for trait, metrics in test_metrics.items():
        print(f"{result['model']} {trait}: test pearson={metrics['pearson']:.6f}, rmse={metrics['rmse']:.6f}", flush=True)


if __name__ == "__main__":
    main()
