from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SPLIT_LABELS = ("train", "valid", "test")
DEFAULT_GBLUP_LAMBDAS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)


def normalize_id(value: object) -> str:
    """Return the same stable ID representation used by the preprocessing script.

    Spreadsheet readers often convert integer-like IDs into values such as
    ``1234.0``. Genomic prediction is extremely sensitive to sample alignment,
    so every table is normalized before joining phenotypes to genotypes.
    """
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        number = float(text)
    except ValueError:
        return text[:-2] if text.endswith(".0") else text
    if math.isfinite(number) and number.is_integer():
        return str(int(number))
    return text


def set_random_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch when PyTorch is available."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def read_table(path: Path) -> pd.DataFrame:
    """Read a CSV/TSV/whitespace table with a practical default delimiter.

    Processed files in this repository are CSV files, but raw phenotype files
    may be whitespace-delimited. This helper keeps the command line interface
    flexible without adding dataset-specific branches to the modeling code.
    """
    suffix = path.suffix.lower()
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t")
    if suffix in {".ped", ".map"}:
        return pd.read_csv(path, sep=r"\s+")
    if suffix == ".txt":
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            first_line = handle.readline()
        if "," in first_line:
            return pd.read_csv(path)
        return pd.read_csv(path, sep=r"\s+")
    return pd.read_csv(path)


def write_json(payload: dict, path: Path) -> None:
    """Write JSON with stable indentation for easy inspection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return Pearson correlation, the most common accuracy metric in GS papers.

    A correlation is undefined when either vector is constant. In that case the
    function returns NaN instead of raising, so result files are still written.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2 or np.nanstd(y_true) == 0.0 or np.nanstd(y_pred) == 0.0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute the standard regression metrics used to compare baselines."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "n": int(len(y_true)),
        "pearson": pearson_corr(y_true, y_pred),
        "rmse": float(math.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan"),
    }


@dataclass
class SplitData:
    """Aligned genotype and phenotype arrays for one train/valid/test split."""

    ids: np.ndarray
    X: np.ndarray
    y: np.ndarray


@dataclass
class DatasetBundle:
    """All data needed by the baseline runner."""

    train: SplitData
    valid: SplitData | None
    test: SplitData | None
    marker_names: list[str]
    trait_name: str
    target_column: str


class MarkerPreprocessor:
    """Train-only marker imputation and optional standardization.

    All marker statistics are estimated from the training set only. Validation
    and test samples are transformed with the same means and standard deviations,
    which prevents information leakage from future animals into model training.
    """

    def __init__(self, standardize: bool = True) -> None:
        self.standardize = standardize
        self.marker_means_: np.ndarray | None = None
        self.marker_stds_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> MarkerPreprocessor:
        X = np.asarray(X, dtype=np.float32)
        means = np.nanmean(X, axis=0)
        means = np.where(np.isfinite(means), means, 0.0).astype(np.float32)
        self.marker_means_ = means

        if self.standardize:
            filled = np.where(np.isnan(X), means, X)
            stds = filled.std(axis=0)
            stds = np.where(stds > 1e-8, stds, 1.0).astype(np.float32)
        else:
            stds = np.ones(X.shape[1], dtype=np.float32)
        self.marker_stds_ = stds
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.marker_means_ is None or self.marker_stds_ is None:
            raise RuntimeError("MarkerPreprocessor must be fitted before transform().")
        X = np.asarray(X, dtype=np.float32)
        X = np.where(np.isnan(X), self.marker_means_, X)
        if self.standardize:
            X = (X - self.marker_means_) / self.marker_stds_
        return X.astype(np.float32, copy=False)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def read_genotype_matrix(path: Path, id_col: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Read a wide genotype dosage matrix into IDs, marker values, and marker names.

    The expected format is one row per animal and one column per marker, with an
    ID column. Marker values should be coded as allele dosages 0/1/2. Missing
    values are allowed and are imputed later using training marker means.
    """
    df = read_table(path)
    if id_col not in df.columns:
        raise ValueError(f"Genotype file {path} does not contain ID column {id_col!r}.")

    ids = df[id_col].map(normalize_id).to_numpy(dtype=str)
    marker_names = [column for column in df.columns if column != id_col]
    if not marker_names:
        raise ValueError(f"Genotype file {path} has no marker columns.")

    # Converting column-by-column avoids pandas object arrays when some markers
    # were read as strings because of missing-value tokens.
    marker_df = df[marker_names].apply(pd.to_numeric, errors="coerce")
    X = marker_df.to_numpy(dtype=np.float32, copy=True)
    return ids, X, marker_names


def load_labels(
    path: Path,
    id_col: str,
    target_column: str,
    trait: str | None,
    trait_column: str,
    split_column: str,
) -> pd.DataFrame:
    """Load phenotype/EBV labels and reduce repeated records to one value per ID.

    Most genomic selection benchmarks predict one phenotype, corrected phenotype,
    deregressed EBV, or EBV per animal. If a long-form file contains repeated
    records for the same animal and trait, the default behavior is to average
    them within ID. This is a conservative baseline choice and avoids sample
    leakage from repeated records of the same animal.
    """
    labels = read_table(path)
    if id_col not in labels.columns:
        raise ValueError(f"Label file {path} does not contain ID column {id_col!r}.")
    if target_column not in labels.columns:
        raise ValueError(f"Label file {path} does not contain target column {target_column!r}.")

    labels = labels.copy()
    labels[id_col] = labels[id_col].map(normalize_id)
    labels[target_column] = pd.to_numeric(labels[target_column], errors="coerce")
    labels = labels[(labels[id_col] != "") & labels[target_column].notna()].copy()

    if trait is not None:
        if trait_column not in labels.columns:
            raise ValueError(f"Trait filtering requested, but {trait_column!r} is missing from {path}.")
        labels = labels[labels[trait_column].astype(str) == str(trait)].copy()
        if labels.empty:
            raise ValueError(f"No label rows remain after filtering {trait_column} == {trait!r}.")

    # Preserve split labels when they are present. The genotype split files also
    # define splits, so this column is optional in that mode.
    group_cols = [id_col]
    if split_column in labels.columns:
        group_cols.append(split_column)

    labels = labels.groupby(group_cols, as_index=False).agg({target_column: "mean"}).rename(columns={id_col: "ID"})
    return labels


def align_split(
    ids: np.ndarray,
    X: np.ndarray,
    labels: pd.DataFrame,
    target_column: str,
    split_name: str | None = None,
    split_column: str = "split",
) -> SplitData:
    """Align a genotype matrix to labels by ID and return matched arrays.

    Genotype order and phenotype order often differ. The safest strategy is to
    build an ID-to-row lookup and explicitly gather matching rows in label order.
    """
    label_df = labels.copy()
    if split_name is not None and split_column in label_df.columns:
        label_df = label_df[label_df[split_column] == split_name].copy()
    if label_df.empty:
        raise ValueError(f"No labels are available for split {split_name!r}.")

    row_by_id = {sample_id: idx for idx, sample_id in enumerate(ids)}
    keep_ids: list[str] = []
    row_indices: list[int] = []
    y_values: list[float] = []
    for sample_id_raw, target_value in label_df[["ID", target_column]].to_numpy():
        sample_id = normalize_id(sample_id_raw)
        if sample_id in row_by_id:
            keep_ids.append(sample_id)
            row_indices.append(row_by_id[sample_id])
            y_values.append(float(target_value))

    if not row_indices:
        raise ValueError(f"No overlapping genotype-label IDs were found for split {split_name!r}.")
    return SplitData(
        ids=np.asarray(keep_ids, dtype=str),
        X=X[np.asarray(row_indices, dtype=int)],
        y=np.asarray(y_values, dtype=np.float32),
    )


def load_dataset_bundle(
    genotype_train: Path | None,
    genotype_valid: Path | None,
    genotype_test: Path | None,
    genotypes: Path | None,
    labels_path: Path,
    id_col: str,
    target_column: str,
    trait: str | None,
    trait_column: str,
    split_column: str,
) -> DatasetBundle:
    """Load split genotype files or a single genotype file plus split labels."""
    labels = load_labels(labels_path, id_col, target_column, trait, trait_column, split_column)
    trait_name = trait if trait is not None else target_column

    if genotype_train is not None:
        train_ids, X_train, marker_names = read_genotype_matrix(genotype_train, id_col)
        train = align_split(train_ids, X_train, labels, target_column, "train", split_column)

        valid = None
        if genotype_valid is not None:
            valid_ids, X_valid, valid_markers = read_genotype_matrix(genotype_valid, id_col)
            if valid_markers != marker_names:
                raise ValueError("Validation genotype markers do not match training markers.")
            valid = align_split(valid_ids, X_valid, labels, target_column, "valid", split_column)

        test = None
        if genotype_test is not None:
            test_ids, X_test, test_markers = read_genotype_matrix(genotype_test, id_col)
            if test_markers != marker_names:
                raise ValueError("Test genotype markers do not match training markers.")
            test = align_split(test_ids, X_test, labels, target_column, "test", split_column)

        return DatasetBundle(train=train, valid=valid, test=test, marker_names=marker_names, trait_name=trait_name, target_column=target_column)

    if genotypes is None:
        raise ValueError("Provide either --genotype-train or --genotypes.")
    if split_column not in labels.columns:
        raise ValueError("A single --genotypes file requires labels with a split column.")

    ids, X, marker_names = read_genotype_matrix(genotypes, id_col)
    train = align_split(ids, X, labels, target_column, "train", split_column)
    valid = align_split(ids, X, labels, target_column, "valid", split_column) if (labels[split_column] == "valid").any() else None
    test = align_split(ids, X, labels, target_column, "test", split_column) if (labels[split_column] == "test").any() else None
    return DatasetBundle(train=train, valid=valid, test=test, marker_names=marker_names, trait_name=trait_name, target_column=target_column)


class GBLUPRegressor:
    """GBLUP implemented as kernel ridge regression with a VanRaden G matrix.

    Genotypes are assumed to be allele dosages coded 0, 1, and 2. The genomic
    relationship between animals i and j is:

        G_ij = (M_i - 2p)'(M_j - 2p) / (2 * sum_k p_k * (1 - p_k))

    where p_k is the training allele frequency of marker k. Predictions for new
    animals use the cross-relationship matrix between new animals and training
    animals. The ridge term lambda approximates the residual-to-genetic variance
    ratio; in practice it is often selected by validation accuracy.
    """

    def __init__(self, lambda_ratio: float = 1.0, jitter: float = 1e-6) -> None:
        self.lambda_ratio = float(lambda_ratio)
        self.jitter = float(jitter)
        self.marker_means_: np.ndarray | None = None
        self.allele_freq_: np.ndarray | None = None
        self.denominator_: float | None = None
        self.W_train_: np.ndarray | None = None
        self.y_mean_: float | None = None
        self.alpha_: np.ndarray | None = None

    def _center_markers(self, X: np.ndarray, fit: bool) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if fit:
            means = np.nanmean(X, axis=0)
            means = np.where(np.isfinite(means), means, 0.0).astype(np.float32)
            self.marker_means_ = means
            self.allele_freq_ = np.clip(means / 2.0, 1e-6, 1.0 - 1e-6).astype(np.float32)
            denom = 2.0 * float(np.sum(self.allele_freq_ * (1.0 - self.allele_freq_)))
            self.denominator_ = denom if denom > 1e-8 else float(X.shape[1])

        if self.marker_means_ is None or self.allele_freq_ is None:
            raise RuntimeError("GBLUPRegressor must be fitted before calling predict().")

        X_filled = np.where(np.isnan(X), self.marker_means_, X)
        return (X_filled - 2.0 * self.allele_freq_).astype(np.float32, copy=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> GBLUPRegressor:
        W = self._center_markers(X, fit=True)
        y = np.asarray(y, dtype=np.float64)
        self.y_mean_ = float(y.mean())
        y_centered = y - self.y_mean_

        if self.denominator_ is None:
            raise RuntimeError("GBLUP denominator was not initialized.")
        K = (W @ W.T).astype(np.float64) / self.denominator_
        K.flat[:: K.shape[0] + 1] += self.lambda_ratio + self.jitter

        # Solving the mixed-model/kernel system is more stable than explicitly
        # inverting K. If the matrix is numerically singular, fall back to the
        # Moore-Penrose pseudo-inverse so the baseline can still run.
        try:
            alpha = np.linalg.solve(K, y_centered)
        except np.linalg.LinAlgError:
            alpha = np.linalg.pinv(K) @ y_centered

        self.W_train_ = W
        self.alpha_ = alpha.astype(np.float64)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.W_train_ is None or self.alpha_ is None or self.y_mean_ is None or self.denominator_ is None:
            raise RuntimeError("GBLUPRegressor must be fitted before predict().")
        W = self._center_markers(X, fit=False)
        K_cross = (W @ self.W_train_.T).astype(np.float64) / self.denominator_
        return (self.y_mean_ + K_cross @ self.alpha_).astype(np.float32)


def choose_best_gblup(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray | None,
    y_valid: np.ndarray | None,
    lambdas: Iterable[float] = DEFAULT_GBLUP_LAMBDAS,
) -> tuple[GBLUPRegressor, dict[str, float]]:
    """Fit GBLUP models over lambda values and keep the best validation model."""
    if X_valid is None or y_valid is None:
        model = GBLUPRegressor(lambda_ratio=1.0).fit(X_train, y_train)
        return model, {"lambda_ratio": 1.0, "selection_rule": "default_no_validation"}

    best_model: GBLUPRegressor | None = None
    best_score = -np.inf
    best_info: dict[str, float] = {}

    for lambda_ratio in lambdas:
        model = GBLUPRegressor(lambda_ratio=lambda_ratio).fit(X_train, y_train)
        pred = model.predict(X_valid)
        metrics = regression_metrics(y_valid, pred)
        metrics["lambda_ratio"] = float(lambda_ratio)
        score = metrics["pearson"]
        if np.isnan(score):
            score = -metrics["rmse"]

        if score > best_score:
            best_score = score
            best_model = model
            best_info = metrics

    if best_model is None:
        raise RuntimeError("No GBLUP model was fitted.")
    return best_model, best_info


def make_svr() -> Pipeline:
    """Create the SVR baseline commonly used in genomic prediction studies.

    Linear SVR is used here because livestock SNP matrices are very wide. It is
    a standard support-vector baseline for high-dimensional genomic prediction
    and scales much better than an RBF-kernel SVR on tens of thousands of SNPs.
    Standardization is important because support-vector objectives depend on
    feature scale.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("model", LinearSVR(C=1.0, epsilon=0.1, max_iter=10000, random_state=42)),
        ]
    )


def make_random_forest(seed: int, n_jobs: int) -> Pipeline:
    """Create a Random Forest baseline for high-dimensional SNP predictors.

    Tree ensembles do not require standardized markers. ``max_features='sqrt'``
    is a common high-dimensional default because each split considers only a
    subset of markers, reducing correlation among trees and training cost.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=500,
                    max_features="sqrt",
                    min_samples_leaf=1,
                    random_state=seed,
                    n_jobs=n_jobs,
                ),
            ),
        ]
    )


def make_random_forest_with_params(
    seed: int,
    n_jobs: int,
    n_estimators: int = 500,
    max_features: str | float = "sqrt",
    min_samples_leaf: int = 1,
    max_depth: int | None = None,
    min_samples_split: int = 2,
) -> Pipeline:
    """Create a Random Forest baseline with tunable regularization knobs."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_features=max_features,
                    min_samples_leaf=min_samples_leaf,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=seed,
                    n_jobs=n_jobs,
                ),
            ),
        ]
    )


def make_xgboost(seed: int, n_jobs: int, device: str = "cpu"):
    """Create an XGBoost regressor with conservative genomic-prediction defaults."""
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError("xgboost is required for the XGBoost baseline. Install it with `pip install xgboost`.") from exc

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            (
                "model",
                XGBRegressor(
                    objective="reg:squarederror",
                    eval_metric="rmse",
                    n_estimators=800,
                    learning_rate=0.03,
                    max_depth=4,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    reg_alpha=0.0,
                    tree_method="hist",
                    device=device,
                    random_state=seed,
                    n_jobs=n_jobs,
                ),
            ),
        ]
    )


@dataclass
class TorchTrainingConfig:
    """Hyperparameters shared by CNN and RNN baselines."""

    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 15
    device: str = "auto"
    seed: int = 42
    selection_metric: str = "loss"
    cnn_dropout: float = 0.2
    rnn_block_size: int = 64
    rnn_hidden_size: int = 64
    rnn_dropout: float = 0.2
    standardize_markers: bool = True


def get_torch_device(device: str):
    """Resolve the requested PyTorch device."""
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class GenotypeArrayDataset:
    """Minimal PyTorch dataset that stores marker arrays and standardized targets."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        import torch

        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
    """Build a DataLoader lazily so PyTorch remains an optional dependency."""
    from torch.utils.data import DataLoader

    return DataLoader(GenotypeArrayDataset(X, y), batch_size=batch_size, shuffle=shuffle, drop_last=False)


class CNNGenomicRegressor:
    """One-dimensional CNN for ordered marker dosages.

    The model treats the SNP vector as a one-channel signal. Convolutional
    filters learn short genomic patterns, while adaptive average pooling keeps
    the final layer independent of the number of markers.
    """

    def __init__(self, n_markers: int, dropout: float = 0.2) -> None:
        import torch.nn as nn

        self.model = nn.Sequential(
            nn.Unflatten(1, (1, n_markers)),
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def __call__(self):
        return self.model


class RNNGenomicRegressor:
    """GRU-based recurrent baseline for marker sequences.

    Feeding tens of thousands of SNPs directly to an RNN is usually too slow.
    This baseline first compresses consecutive markers into fixed-size blocks,
    then sends the block sequence to a bidirectional GRU. The design preserves
    marker order while keeping the recurrent sequence length manageable.
    """

    def __init__(self, n_markers: int, block_size: int = 64, hidden_size: int = 64, dropout: float = 0.2) -> None:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class _BlockGRU(nn.Module):
            def __init__(self, n_markers: int, block_size: int, hidden_size: int, dropout: float) -> None:
                super().__init__()
                self.n_markers = n_markers
                self.block_size = block_size
                self.n_blocks = int(math.ceil(n_markers / block_size))
                self.padded_markers = self.n_blocks * block_size
                self.block_projection = nn.Linear(block_size, hidden_size)
                self.gru = nn.GRU(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                )
                self.head = nn.Sequential(
                    nn.LayerNorm(hidden_size * 2),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size * 2, 1),
                )

            def forward(self, x):
                if self.padded_markers > self.n_markers:
                    x = F.pad(x, (0, self.padded_markers - self.n_markers))
                x = x.view(x.shape[0], self.n_blocks, self.block_size)
                x = torch.relu(self.block_projection(x))
                _, hidden = self.gru(x)
                hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
                return self.head(hidden)

        self.model = _BlockGRU(n_markers, block_size, hidden_size, dropout)

    def __call__(self):
        return self.model


class TorchRegressor:
    """Scikit-learn-like wrapper around a PyTorch regression model."""

    def __init__(
        self,
        model_factory: Callable[[int], object],
        config: TorchTrainingConfig,
        standardize_markers: bool = True,
    ) -> None:
        self.model_factory = model_factory
        self.config = config
        self.standardize_markers = standardize_markers
        self.preprocessor_: MarkerPreprocessor | None = None
        self.y_mean_: float | None = None
        self.y_std_: float | None = None
        self.model_ = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: np.ndarray | None = None,
        y_valid: np.ndarray | None = None,
    ) -> TorchRegressor:
        import torch
        import torch.nn as nn

        set_random_seed(self.config.seed)
        device = get_torch_device(self.config.device)

        self.preprocessor_ = MarkerPreprocessor(standardize=self.standardize_markers)
        X_train_p = self.preprocessor_.fit_transform(X_train)
        X_valid_p = self.preprocessor_.transform(X_valid) if X_valid is not None else None

        self.y_mean_ = float(np.mean(y_train))
        y_std = float(np.std(y_train))
        self.y_std_ = y_std if y_std > 1e-8 else 1.0
        y_train_z = ((y_train - self.y_mean_) / self.y_std_).astype(np.float32)
        y_valid_z = ((y_valid - self.y_mean_) / self.y_std_).astype(np.float32) if y_valid is not None else None

        self.model_ = self.model_factory(X_train_p.shape[1]).to(device)
        if device.type == "cuda" and torch.cuda.device_count() > 1:
            # DataParallel lets the neural baselines use both local GPUs while
            # keeping the same single-process training interface.
            self.model_ = nn.DataParallel(self.model_)
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        criterion = nn.MSELoss()
        train_loader = make_loader(X_train_p, y_train_z, self.config.batch_size, shuffle=True)
        valid_loader = make_loader(X_valid_p, y_valid_z, self.config.batch_size, shuffle=False) if X_valid_p is not None else None

        best_state = None
        best_monitor = -np.inf if self.config.selection_metric == "pearson" else float("inf")
        epochs_without_improvement = 0

        for _epoch in range(1, self.config.epochs + 1):
            self.model_.train()
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.model_(xb), yb)
                loss.backward()
                optimizer.step()

            if valid_loader is None:
                continue

            self.model_.eval()
            valid_losses: list[float] = []
            with torch.no_grad():
                for xb, yb in valid_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    valid_losses.append(float(criterion(self.model_(xb), yb).item()))
            valid_loss = float(np.mean(valid_losses))
            monitor = valid_loss
            improved = valid_loss + 1e-6 < best_monitor

            if self.config.selection_metric == "pearson":
                valid_pred_batches: list[np.ndarray] = []
                with torch.no_grad():
                    for xb, _ in valid_loader:
                        xb = xb.to(device)
                        valid_pred_batches.append(self.model_(xb).detach().cpu().numpy().reshape(-1))
                valid_pred_z = np.concatenate(valid_pred_batches)
                valid_pcc = pearson_corr(y_valid_z, valid_pred_z)
                monitor = valid_pcc if np.isfinite(valid_pcc) else -np.inf
                improved = monitor > best_monitor + 1e-6

            if improved:
                best_monitor = monitor
                best_state = {key: value.detach().cpu().clone() for key, value in self.model_.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.config.patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch

        if self.preprocessor_ is None or self.model_ is None or self.y_mean_ is None or self.y_std_ is None:
            raise RuntimeError("TorchRegressor must be fitted before predict().")
        device = next(self.model_.parameters()).device
        X_p = self.preprocessor_.transform(X)
        loader = make_loader(X_p, np.zeros(X_p.shape[0], dtype=np.float32), self.config.batch_size, shuffle=False)
        preds: list[np.ndarray] = []
        self.model_.eval()
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(device)
                pred = self.model_(xb).detach().cpu().numpy().reshape(-1)
                preds.append(pred)
        pred_z = np.concatenate(preds)
        return (pred_z * self.y_std_ + self.y_mean_).astype(np.float32)


def fit_classical_model(
    name: str,
    bundle: DatasetBundle,
    seed: int,
    n_jobs: int,
    xgboost_device: str = "cpu",
    rf_n_estimators: int = 500,
    rf_max_features: str = "sqrt",
    rf_min_samples_leaf: int = 1,
    rf_max_depth: int = 0,
    rf_min_samples_split: int = 2,
):
    """Fit one non-deep-learning baseline and return predictions plus metadata."""
    X_train, y_train = bundle.train.X, bundle.train.y
    X_valid = bundle.valid.X if bundle.valid is not None else None
    y_valid = bundle.valid.y if bundle.valid is not None else None

    if name == "gblup":
        model, validation_info = choose_best_gblup(X_train, y_train, X_valid, y_valid)
        return model, {"validation_selection": validation_info}
    if name == "svr":
        model = make_svr().fit(X_train, y_train)
        return model, {}
    if name == "rf":
        max_depth = None if rf_max_depth <= 0 else rf_max_depth
        max_features: str | float = rf_max_features
        try:
            if rf_max_features not in {"sqrt", "log2"}:
                max_features = float(rf_max_features)
        except ValueError:
            max_features = rf_max_features
        model = make_random_forest_with_params(
            seed,
            n_jobs,
            n_estimators=rf_n_estimators,
            max_features=max_features,
            min_samples_leaf=rf_min_samples_leaf,
            max_depth=max_depth,
            min_samples_split=rf_min_samples_split,
        ).fit(X_train, y_train)
        return model, {}
    if name == "xgboost":
        model = make_xgboost(seed, n_jobs, xgboost_device).fit(X_train, y_train)
        return model, {}
    raise ValueError(f"Unknown classical baseline {name!r}.")


def fit_torch_model(name: str, bundle: DatasetBundle, config: TorchTrainingConfig) -> TorchRegressor:
    """Fit one deep-learning baseline."""
    if name == "cnn":
        factory = lambda n_markers: CNNGenomicRegressor(n_markers, dropout=config.cnn_dropout).__call__()
    elif name == "rnn":
        factory = lambda n_markers: RNNGenomicRegressor(
            n_markers,
            block_size=config.rnn_block_size,
            hidden_size=config.rnn_hidden_size,
            dropout=config.rnn_dropout,
        ).__call__()
    else:
        raise ValueError(f"Unknown deep-learning baseline {name!r}.")

    model = TorchRegressor(factory, config, standardize_markers=config.standardize_markers)
    model.fit(
        bundle.train.X,
        bundle.train.y,
        bundle.valid.X if bundle.valid is not None else None,
        bundle.valid.y if bundle.valid is not None else None,
    )
    return model


def evaluate_model(name: str, model, bundle: DatasetBundle, extra: dict | None = None) -> dict:
    """Evaluate a fitted model on train/valid/test and return serializable output."""
    result: dict = {
        "model": name,
        "trait": bundle.trait_name,
        "target_column": bundle.target_column,
        "n_markers": len(bundle.marker_names),
        "metrics": {},
    }
    if extra:
        result.update(extra)

    for split_name in SPLIT_LABELS:
        split_data = getattr(bundle, split_name)
        if split_data is None:
            continue
        pred = model.predict(split_data.X)
        result["metrics"][split_name] = regression_metrics(split_data.y, pred)
    return result


def save_predictions(name: str, model, bundle: DatasetBundle, output_dir: Path) -> None:
    """Write per-sample predictions for downstream plots and error analysis."""
    rows = []
    for split_name in SPLIT_LABELS:
        split_data = getattr(bundle, split_name)
        if split_data is None:
            continue
        pred = model.predict(split_data.X)
        rows.append(
            pd.DataFrame(
                {
                    "ID": split_data.ids,
                    "split": split_name,
                    "y_true": split_data.y,
                    "y_pred": pred,
                    "model": name,
                    "trait": bundle.trait_name,
                }
            )
        )
    if rows:
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.concat(rows, ignore_index=True).to_csv(output_dir / f"predictions_{name}.csv", index=False)


def parse_models(values: list[str]) -> list[str]:
    """Expand the all shortcut and keep model order stable."""
    all_models = ["gblup", "svr", "rf", "xgboost", "cnn", "rnn"]
    if "all" in values:
        return all_models
    unknown = sorted(set(values) - set(all_models))
    if unknown:
        raise ValueError(f"Unknown model names: {unknown}. Valid choices are {all_models} or all.")
    return values


def parse_args() -> argparse.Namespace:
    """Parse command-line options for training genomic selection baselines."""
    parser = argparse.ArgumentParser(
        description=(
            "Train genomic selection baselines: GBLUP, SVR, Random Forest, "
            "XGBoost, CNN, and RNN."
        )
    )
    genotype_group = parser.add_mutually_exclusive_group(required=True)
    genotype_group.add_argument("--genotypes", type=Path, help="Single genotype matrix with ID and marker columns.")
    genotype_group.add_argument(
        "--genotype-dir",
        type=Path,
        help="Directory containing genotypes_train.csv, genotypes_valid.csv, and genotypes_test.csv.",
    )
    genotype_group.add_argument("--genotype-train", type=Path, help="Training genotype matrix with ID and marker columns.")
    parser.add_argument("--genotype-valid", type=Path, help="Validation genotype matrix. Used with --genotype-train.")
    parser.add_argument("--genotype-test", type=Path, help="Test genotype matrix. Used with --genotype-train.")
    parser.add_argument("--labels", type=Path, required=True, help="Phenotype or EBV label table.")
    parser.add_argument("--trait", type=str, help="Trait name to select from a long-form label table.")
    parser.add_argument("--trait-column", type=str, default="trait", help="Column containing trait names.")
    parser.add_argument("--target-column", type=str, default="value", help="Column containing numeric targets.")
    parser.add_argument("--id-column", type=str, default="ID", help="Sample/animal ID column.")
    parser.add_argument("--split-column", type=str, default="split", help="Split label column in the label table.")
    parser.add_argument("--models", nargs="+", default=["all"], help="Subset of: gblup svr rf xgboost cnn rnn all.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/baselines"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--selection-metric", type=str, choices=["loss", "pearson"], default="loss")
    parser.add_argument("--cnn-dropout", type=float, default=0.2)
    parser.add_argument("--rnn-block-size", type=int, default=64)
    parser.add_argument("--rnn-hidden-size", type=int, default=64)
    parser.add_argument("--rnn-dropout", type=float, default=0.2)
    parser.add_argument("--no-standardize-markers", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", type=str, default="auto", help="PyTorch device: auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--xgboost-device", type=str, default="cpu", help="XGBoost device: cpu, cuda, cuda:0, etc.")
    parser.add_argument("--rf-n-estimators", type=int, default=500)
    parser.add_argument("--rf-max-features", type=str, default="sqrt")
    parser.add_argument("--rf-min-samples-leaf", type=int, default=1)
    parser.add_argument("--rf-max-depth", type=int, default=0, help="<=0 means unlimited depth.")
    parser.add_argument("--rf-min-samples-split", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    """Entry point used by the command-line script."""
    args = parse_args()
    set_random_seed(args.seed)
    models = parse_models(args.models)
    genotype_train = args.genotype_train
    genotype_valid = args.genotype_valid
    genotype_test = args.genotype_test
    if args.genotype_dir is not None:
        genotype_train = args.genotype_dir / "genotypes_train.csv"
        genotype_valid = args.genotype_dir / "genotypes_valid.csv"
        genotype_test = args.genotype_dir / "genotypes_test.csv"

    bundle = load_dataset_bundle(
        genotype_train=genotype_train,
        genotype_valid=genotype_valid,
        genotype_test=genotype_test,
        genotypes=args.genotypes,
        labels_path=args.labels,
        id_col=args.id_column,
        target_column=args.target_column,
        trait=args.trait,
        trait_column=args.trait_column,
        split_column=args.split_column,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    torch_config = TorchTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=args.device,
        seed=args.seed,
        selection_metric=args.selection_metric,
        cnn_dropout=args.cnn_dropout,
        rnn_block_size=args.rnn_block_size,
        rnn_hidden_size=args.rnn_hidden_size,
        rnn_dropout=args.rnn_dropout,
        standardize_markers=not args.no_standardize_markers,
    )

    for name in models:
        print(f"Training {name}...")
        if name in {"cnn", "rnn"}:
            model = fit_torch_model(name, bundle, torch_config)
            extra = {"torch_config": torch_config.__dict__}
        else:
            model, extra = fit_classical_model(
                name,
                bundle,
                args.seed,
                args.n_jobs,
                args.xgboost_device,
                rf_n_estimators=args.rf_n_estimators,
                rf_max_features=args.rf_max_features,
                rf_min_samples_leaf=args.rf_min_samples_leaf,
                rf_max_depth=args.rf_max_depth,
                rf_min_samples_split=args.rf_min_samples_split,
            )

        result = evaluate_model(name, model, bundle, extra=extra)
        save_predictions(name, model, bundle, args.output_dir)
        write_json(result, args.output_dir / f"metrics_{name}.json")
        all_results.append(result)

        if "valid" in result["metrics"]:
            valid = result["metrics"]["valid"]
            print(f"{name}: valid pearson={valid['pearson']:.4f}, valid rmse={valid['rmse']:.4f}")
        elif "test" in result["metrics"]:
            test = result["metrics"]["test"]
            print(f"{name}: test pearson={test['pearson']:.4f}, test rmse={test['rmse']:.4f}")

    write_json({"results": all_results}, args.output_dir / "metrics_all.json")


if __name__ == "__main__":
    main()
