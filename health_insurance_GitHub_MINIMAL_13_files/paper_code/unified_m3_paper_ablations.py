#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified, development-only ablation suite for the health-insurance paper.

The script answers four paper questions under one controlled protocol:

1. Initial learning-rate sensitivity on M3.
2. Shared MTL versus two separate MLPs on M3.
3. Standardized log1p-MSE versus direct Gamma-mean severity on M3.
4. M3 versus the proposed M4 representation under the frozen Gamma-MTL.

No 2018->2019 file is referenced or loaded. Every experiment uses the same
2017->2018 cohort, policy-grouped 80/20 split, train-fitted preprocessing,
four seeds, 35 epochs, batch size 512, AdamW, weight decay, scheduler and
evaluation definitions. Results are intended to be interpreted within each
ablation block, not as an exhaustive factorial search.

Paper profile:
    python -u unified_m3_paper_ablations.py --profile paper --stage all

Quick integrity check:
    python -u unified_m3_paper_ablations.py --profile quick --stage all
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------------------------------------------------------
# Frozen scientific protocol
# -----------------------------------------------------------------------------

DEFAULT_PROJECT_DIR = Path(
    os.environ.get("HEALTH_PROJECT_DIR", "/home/eleut/health_insurance_project")
).expanduser()

PROFILES: Dict[str, Dict[str, Any]] = {
    "quick": {"seeds": [42], "epochs": 3, "batch_size": 512},
    "paper": {
        "seeds": [42, 123, 2026, 31415],
        "epochs": 35,
        "batch_size": 512,
    },
}

VALIDATION_SIZE = 0.20
SPLIT_SEED = 42
REFERENCE_LR = 1e-3
LEARNING_RATES = [1e-4, 1e-3, 1e-2, 1e-1]
LOG_MSE_WEIGHT = 0.50
GAMMA_WEIGHT = 0.10
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 5.0
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 3
SCHEDULER_MIN_LR = 1e-7
MIN_LOG_MEAN = -8.0
MAX_LOG_MEAN = 8.0
EPS = 1e-8

BASE_NUMERIC = [
    "age",
    "exposure_time",
    "seniority_insured",
    "seniority_policy",
    "log_premium",
]
HISTORY_NUMERIC = ["prev_occurrence", "log_prev_cost", "log_prev_services"]
BASE_CATEGORICAL = [
    "gender",
    "type_policy",
    "type_policy_dg",
    "type_product",
    "reimbursement",
    "new_business",
    "distribution_channel",
]
CONTEXT_NUMERIC = [
    "n_insured_pc",
    "n_insured_mun",
    "n_insured_prov",
    "IICIMUN",
    "IICIPROV",
    "C_GI",
    "C_II",
    "C_GE_P",
    "C_GE_S",
    "C_GE_T",
    "C_IE_P",
    "C_IE_S",
    "C_IE_T",
]
CONTEXT_CATEGORICAL = ["C_H", "C_C"]
M4_ADDED_NUMERIC = [
    "m4_log_prev_loss_ratio",
    "m4_log_prev_cost_per_service",
    "m4_cost_intensity",
    "m4_age_prev_occurrence",
    "m4_services_prev_occurrence",
    "m4_relative_log_premium_product",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=DEFAULT_PROJECT_DIR)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="paper")
    parser.add_argument(
        "--stage",
        choices=["all", "lr", "architecture", "objective", "representation"],
        default="all",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_cpu() -> torch.device:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return torch.device("cpu")


def save_json(payload: Mapping[str, Any], path: Path) -> None:
    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(type(value).__name__)

    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=convert),
        encoding="utf-8",
    )


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(actual, predicted)))


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# -----------------------------------------------------------------------------
# Data and feature engineering
# -----------------------------------------------------------------------------

def engineer_m3(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    previous_cost = pd.to_numeric(result["cost_claims_year"], errors="coerce")
    previous_services = pd.to_numeric(
        result["n_medical_services"], errors="coerce"
    )
    premium = pd.to_numeric(result["premium"], errors="coerce")
    result["prev_occurrence"] = (previous_cost > 0).astype("float32")
    result["log_prev_cost"] = np.log1p(previous_cost.clip(lower=0)).astype(
        "float32"
    )
    result["log_prev_services"] = np.log1p(
        previous_services.clip(lower=0)
    ).astype("float32")
    result["log_premium"] = np.log1p(premium.clip(lower=0)).astype("float32")
    return result


def fit_m4_context(train: pd.DataFrame) -> Dict[str, Any]:
    log_premium = pd.to_numeric(train["log_premium"], errors="coerce")
    global_median = float(log_premium.median())
    products = train["type_product"].astype("string").fillna("Missing")
    medians = pd.DataFrame(
        {"product": products, "log_premium": log_premium}
    ).groupby("product", dropna=False)["log_premium"].median()
    return {
        "global_log_premium_median": global_median,
        "product_log_premium_medians": {
            str(key): float(value)
            for key, value in medians.items()
            if pd.notna(value)
        },
    }


def apply_m4(frame: pd.DataFrame, context: Mapping[str, Any]) -> pd.DataFrame:
    result = engineer_m3(frame)
    previous_cost = pd.to_numeric(
        result["cost_claims_year"], errors="coerce"
    ).clip(lower=0)
    previous_services = pd.to_numeric(
        result["n_medical_services"], errors="coerce"
    ).clip(lower=0)
    premium = pd.to_numeric(result["premium"], errors="coerce").clip(lower=0)

    result["m4_log_prev_loss_ratio"] = np.log1p(
        previous_cost / (premium + 1.0)
    ).astype("float32")
    result["m4_log_prev_cost_per_service"] = np.log1p(
        previous_cost / (previous_services + 1.0)
    ).astype("float32")
    result["m4_cost_intensity"] = (
        result["log_prev_cost"] - result["log_prev_services"]
    ).astype("float32")
    age = pd.to_numeric(result["age"], errors="coerce")
    result["m4_age_prev_occurrence"] = (
        age * result["prev_occurrence"]
    ).astype("float32")
    result["m4_services_prev_occurrence"] = (
        result["log_prev_services"] * result["prev_occurrence"]
    ).astype("float32")

    products = result["type_product"].astype("string").fillna("Missing")
    medians = {
        str(key): float(value)
        for key, value in context["product_log_premium_medians"].items()
    }
    reference = products.map(medians).astype(float).fillna(
        float(context["global_log_premium_median"])
    )
    result["m4_relative_log_premium_product"] = (
        pd.to_numeric(result["log_premium"], errors="coerce") - reference
    ).astype("float32")
    return result


def available_features(
    frame: pd.DataFrame, expected: Sequence[str]
) -> List[str]:
    return [name for name in expected if name in frame.columns]


def feature_definition(frame: pd.DataFrame, block: str) -> Tuple[List[str], List[str]]:
    numeric = BASE_NUMERIC + HISTORY_NUMERIC + available_features(
        frame, CONTEXT_NUMERIC
    )
    categorical = BASE_CATEGORICAL + available_features(
        frame, CONTEXT_CATEGORICAL
    )
    if block == "M4":
        numeric += M4_ADDED_NUMERIC
    return list(dict.fromkeys(numeric)), list(dict.fromkeys(categorical))


def build_preprocessor(
    numeric: Sequence[str], categorical: Sequence[str]
) -> ColumnTransformer:
    try:
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="median", add_indicator=True),
                        ),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(numeric),
            ),
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="constant", fill_value="Missing"),
                        ),
                        ("onehot", one_hot),
                    ]
                ),
                list(categorical),
            ),
        ],
        remainder="drop",
    )


def grouped_split(frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=VALIDATION_SIZE, random_state=SPLIT_SEED
    )
    train, validation = next(
        # Preserve the original ID_policy dtype. GroupShuffleSplit internally
        # orders unique group labels before applying its seeded permutation;
        # casting labels to string changes that order and therefore produces a
        # different (although still leakage-free) split. The authoritative
        # project pipelines pass the raw column, so we do the same here.
        splitter.split(frame, groups=frame["ID_policy"])
    )
    overlap = set(frame.iloc[train]["ID_policy"].astype(str)).intersection(
        set(frame.iloc[validation]["ID_policy"].astype(str))
    )
    if overlap:
        raise RuntimeError("Policy overlap detected in grouped split.")
    return train.astype(int), validation.astype(int)


def prepare_matrix(
    frame: pd.DataFrame,
    train_idx: np.ndarray,
    validation_idx: np.ndarray,
    block: str,
    output_dir: Path,
) -> Dict[str, Any]:
    numeric, categorical = feature_definition(frame, block)
    features = numeric + categorical
    missing = [name for name in features if name not in frame.columns]
    if missing:
        raise KeyError(f"{block} missing features: {missing}")
    preprocessor = build_preprocessor(numeric, categorical)
    X_train = np.asarray(
        preprocessor.fit_transform(frame.iloc[train_idx][features]),
        dtype=np.float32,
    )
    X_validation = np.asarray(
        preprocessor.transform(frame.iloc[validation_idx][features]),
        dtype=np.float32,
    )
    if not np.isfinite(X_train).all() or not np.isfinite(X_validation).all():
        raise ValueError(f"{block} encoded matrix contains non-finite values.")
    joblib.dump(preprocessor, output_dir / f"{block.lower()}_preprocessor.joblib")
    return {
        "block": block,
        "numeric": numeric,
        "categorical": categorical,
        "features": features,
        "input_dim": int(X_train.shape[1]),
        "X_train": X_train,
        "X_validation": X_validation,
    }


# -----------------------------------------------------------------------------
# Models and losses
# -----------------------------------------------------------------------------

class SharedMTL(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
        )
        self.occurrence_head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
        )
        self.severity_head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        representation = self.shared(x)
        return (
            self.occurrence_head(representation).squeeze(1),
            self.severity_head(representation).squeeze(1),
        )


class SingleTaskMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)


def gamma_mean_nll(raw_log_mean: torch.Tensor, scaled_cost: torch.Tensor) -> torch.Tensor:
    log_mean = torch.clamp(raw_log_mean, MIN_LOG_MEAN, MAX_LOG_MEAN)
    mean = torch.exp(log_mean).clamp_min(EPS)
    return torch.mean(scaled_cost / mean + log_mean)


def make_loader(
    X: np.ndarray,
    y_occ: np.ndarray,
    y_sev: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(
            torch.from_numpy(X.astype(np.float32, copy=False)),
            torch.from_numpy(y_occ.astype(np.float32, copy=False)),
            torch.from_numpy(y_sev.astype(np.float32, copy=False)),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=bool(shuffle),
        num_workers=0,
        generator=generator,
    )


def objective_loss(
    occurrence_logit: torch.Tensor,
    severity_raw: torch.Tensor,
    occurrence: torch.Tensor,
    severity_target: torch.Tensor,
    objective: str,
    severity_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    occurrence_loss = nn.functional.binary_cross_entropy_with_logits(
        occurrence_logit, occurrence
    )
    positive = occurrence > 0.5
    if bool(positive.any()):
        if objective == "log_mse":
            severity_loss = nn.functional.mse_loss(
                severity_raw[positive], severity_target[positive]
            )
        elif objective == "gamma":
            severity_loss = gamma_mean_nll(
                severity_raw[positive], severity_target[positive]
            )
        else:
            raise ValueError(objective)
    else:
        severity_loss = severity_raw.sum() * 0.0
    total = occurrence_loss + severity_weight * severity_loss
    return total, occurrence_loss, severity_loss


def evaluate_loss(
    model: SharedMTL,
    loader: DataLoader,
    objective: str,
    severity_weight: float,
    device: torch.device,
) -> float:
    model.eval()
    total_sum = 0.0
    rows = 0
    with torch.no_grad():
        for X, occurrence, severity in loader:
            X, occurrence, severity = X.to(device), occurrence.to(device), severity.to(device)
            occurrence_logit, severity_raw = model(X)
            total, _, _ = objective_loss(
                occurrence_logit,
                severity_raw,
                occurrence,
                severity,
                objective,
                severity_weight,
            )
            total_sum += float(total.item()) * len(X)
            rows += len(X)
    return total_sum / max(rows, 1)


def fit_shared(
    X_train: np.ndarray,
    X_validation: np.ndarray,
    y_occ_train: np.ndarray,
    y_occ_validation: np.ndarray,
    y_sev_train: np.ndarray,
    y_sev_validation: np.ndarray,
    objective: str,
    severity_weight: float,
    learning_rate: float,
    seed: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[SharedMTL, Dict[str, Any]]:
    set_seed(seed)
    train_loader = make_loader(
        X_train, y_occ_train, y_sev_train, batch_size, True, seed
    )
    validation_loader = make_loader(
        X_validation, y_occ_validation, y_sev_validation, 1024, False, seed
    )
    model = SharedMTL(X_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=SCHEDULER_MIN_LR,
    )
    best_loss = math.inf
    best_epoch = -1
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        for X, occurrence, severity in train_loader:
            X, occurrence, severity = X.to(device), occurrence.to(device), severity.to(device)
            optimizer.zero_grad(set_to_none=True)
            occurrence_logit, severity_raw = model(X)
            total, _, _ = objective_loss(
                occurrence_logit,
                severity_raw,
                occurrence,
                severity,
                objective,
                severity_weight,
            )
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
        validation_loss = evaluate_loss(
            model, validation_loader, objective, severity_weight, device
        )
        scheduler.step(validation_loss)
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("No valid shared checkpoint was produced.")
    model.load_state_dict(best_state)
    return model, {
        "best_validation_total_loss": float(best_loss),
        "best_epoch": int(best_epoch),
        "parameters": count_parameters(model),
    }


def fit_single_task(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    learning_rate: float,
    seed: int,
    epochs: int,
    batch_size: int,
    loss_name: str,
    device: torch.device,
) -> Tuple[SingleTaskMLP, Dict[str, Any]]:
    set_seed(seed)
    model = SingleTaskMLP(X_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=SCHEDULER_MIN_LR,
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train.astype(np.float32, copy=False)),
            torch.from_numpy(y_train.astype(np.float32, copy=False)),
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        generator=generator,
    )
    validation_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_validation.astype(np.float32, copy=False)),
            torch.from_numpy(y_validation.astype(np.float32, copy=False)),
        ),
        batch_size=1024,
        shuffle=False,
        num_workers=0,
    )
    best_loss, best_epoch, best_state = math.inf, -1, None
    for epoch in range(1, epochs + 1):
        model.train()
        for X, target in train_loader:
            X, target = X.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(X)
            if loss_name == "bce":
                loss = nn.functional.binary_cross_entropy_with_logits(prediction, target)
            else:
                loss = nn.functional.mse_loss(prediction, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
        model.eval()
        values: List[float] = []
        counts: List[int] = []
        with torch.no_grad():
            for X, target in validation_loader:
                X, target = X.to(device), target.to(device)
                prediction = model(X)
                if loss_name == "bce":
                    loss = nn.functional.binary_cross_entropy_with_logits(prediction, target)
                else:
                    loss = nn.functional.mse_loss(prediction, target)
                values.append(float(loss.item()))
                counts.append(len(X))
        validation_loss = float(np.average(values, weights=counts))
        scheduler.step(validation_loss)
        if validation_loss < best_loss - 1e-7:
            best_loss, best_epoch = validation_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("No valid single-task checkpoint was produced.")
    model.load_state_dict(best_state)
    return model, {
        "best_validation_loss": float(best_loss),
        "best_epoch": int(best_epoch),
        "parameters": count_parameters(model),
    }


# -----------------------------------------------------------------------------
# Prediction and metrics
# -----------------------------------------------------------------------------

def predict_shared(
    model: SharedMTL,
    X: np.ndarray,
    objective: str,
    severity_mean: float,
    severity_std: float,
    gamma_scale: float,
    smearing: float,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities: List[np.ndarray] = []
    severities: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(X), 1024):
            features = torch.from_numpy(X[start : start + 1024]).to(device)
            occurrence_logit, severity_raw = model(features)
            probability = torch.sigmoid(occurrence_logit).cpu().numpy()
            raw = severity_raw.cpu().numpy()
            if objective == "log_mse":
                positive_cost = np.expm1(raw * severity_std + severity_mean)
                positive_cost = positive_cost * smearing
            else:
                positive_cost = np.exp(np.clip(raw, MIN_LOG_MEAN, MAX_LOG_MEAN))
                positive_cost = positive_cost * gamma_scale
            probabilities.append(probability)
            severities.append(np.clip(positive_cost, 0.0, None))
    return np.concatenate(probabilities), np.concatenate(severities)


def predict_single(model: SingleTaskMLP, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    values: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(X), 1024):
            values.append(
                model(torch.from_numpy(X[start : start + 1024]).to(device))
                .cpu()
                .numpy()
            )
    return np.concatenate(values)


def compute_smearing(
    model: SharedMTL,
    X_train: np.ndarray,
    y_log_train: np.ndarray,
    severity_mean: float,
    severity_std: float,
    positive_mask: np.ndarray,
    device: torch.device,
) -> float:
    model.eval()
    raw_values: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(X_train), 1024):
            _, raw = model(torch.from_numpy(X_train[start : start + 1024]).to(device))
            raw_values.append(raw.cpu().numpy())
    predicted_log = np.concatenate(raw_values) * severity_std + severity_mean
    residual = y_log_train[positive_mask] - predicted_log[positive_mask]
    return float(np.mean(np.exp(np.clip(residual, -30.0, 30.0))))


def metrics(
    actual_occurrence: np.ndarray,
    actual_cost: np.ndarray,
    probability: np.ndarray,
    positive_cost: np.ndarray,
) -> Dict[str, float]:
    soft_cost = probability * positive_cost
    positive = actual_occurrence == 1
    return {
        "roc_auc": float(roc_auc_score(actual_occurrence, probability)),
        "brier": float(brier_score_loss(actual_occurrence, probability)),
        "soft_cost_mae": float(mean_absolute_error(actual_cost, soft_cost)),
        "soft_cost_rmse": rmse(actual_cost, soft_cost),
        "severity_rmse": rmse(actual_cost[positive], positive_cost[positive]),
        "portfolio_error_pct": float(
            100.0 * (soft_cost.sum() - actual_cost.sum()) / actual_cost.sum()
        ),
    }


# -----------------------------------------------------------------------------
# Controlled experiment runner
# -----------------------------------------------------------------------------

def run_shared_setting(
    setting: Mapping[str, Any],
    matrix: Mapping[str, Any],
    targets: Mapping[str, np.ndarray],
    cfg: Mapping[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    objective = str(setting["objective"])
    positive_train = targets["occ_train"] == 1
    positive_validation = targets["occ_validation"] == 1
    log_train = np.log1p(targets["cost_train"])
    log_validation = np.log1p(targets["cost_validation"])
    severity_mean = float(log_train[positive_train].mean())
    severity_std = float(log_train[positive_train].std(ddof=0))
    if not np.isfinite(severity_std) or severity_std <= 0:
        raise ValueError("Invalid positive log-cost standard deviation.")
    gamma_scale = float(targets["cost_train"][positive_train].mean())
    if objective == "log_mse":
        sev_train = ((log_train - severity_mean) / severity_std).astype(np.float32)
        sev_validation = ((log_validation - severity_mean) / severity_std).astype(
            np.float32
        )
    else:
        sev_train = (targets["cost_train"] / gamma_scale).astype(np.float32)
        sev_validation = (targets["cost_validation"] / gamma_scale).astype(
            np.float32
        )

    model, training = fit_shared(
        matrix["X_train"],
        matrix["X_validation"],
        targets["occ_train"],
        targets["occ_validation"],
        sev_train,
        sev_validation,
        objective,
        float(setting["severity_weight"]),
        float(setting["learning_rate"]),
        int(setting["seed"]),
        int(cfg["epochs"]),
        int(cfg["batch_size"]),
        device,
    )
    smearing = 1.0
    if objective == "log_mse":
        smearing = compute_smearing(
            model,
            matrix["X_train"],
            log_train,
            severity_mean,
            severity_std,
            positive_train,
            device,
        )
    probability, positive_cost = predict_shared(
        model,
        matrix["X_validation"],
        objective,
        severity_mean,
        severity_std,
        gamma_scale,
        smearing,
        device,
    )
    result = {
        **dict(setting),
        **training,
        **metrics(
            targets["occ_validation"],
            targets["cost_validation"],
            probability,
            positive_cost,
        ),
        "input_dim": int(matrix["input_dim"]),
        "smearing_factor": float(smearing),
        "status": "success",
    }
    del model
    gc.collect()
    return result


def run_separate_setting(
    setting: Mapping[str, Any],
    matrix: Mapping[str, Any],
    targets: Mapping[str, np.ndarray],
    cfg: Mapping[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    positive_train = targets["occ_train"] == 1
    positive_validation = targets["occ_validation"] == 1
    log_train = np.log1p(targets["cost_train"])
    log_validation = np.log1p(targets["cost_validation"])
    severity_mean = float(log_train[positive_train].mean())
    severity_std = float(log_train[positive_train].std(ddof=0))
    standardized_train = (log_train[positive_train] - severity_mean) / severity_std
    standardized_validation = (
        log_validation[positive_validation] - severity_mean
    ) / severity_std

    occurrence_model, occurrence_info = fit_single_task(
        matrix["X_train"],
        targets["occ_train"],
        matrix["X_validation"],
        targets["occ_validation"],
        float(setting["learning_rate"]),
        int(setting["seed"]),
        int(cfg["epochs"]),
        int(cfg["batch_size"]),
        "bce",
        device,
    )
    severity_model, severity_info = fit_single_task(
        matrix["X_train"][positive_train],
        standardized_train.astype(np.float32),
        matrix["X_validation"][positive_validation],
        standardized_validation.astype(np.float32),
        float(setting["learning_rate"]),
        int(setting["seed"]),
        int(cfg["epochs"]),
        int(cfg["batch_size"]),
        "mse",
        device,
    )
    probability = 1.0 / (
        1.0 + np.exp(-np.clip(predict_single(occurrence_model, matrix["X_validation"], device), -40, 40))
    )
    raw_severity = predict_single(severity_model, matrix["X_validation"], device)
    train_raw = predict_single(severity_model, matrix["X_train"][positive_train], device)
    train_predicted_log = train_raw * severity_std + severity_mean
    smearing = float(
        np.mean(np.exp(np.clip(log_train[positive_train] - train_predicted_log, -30, 30)))
    )
    positive_cost = np.clip(
        np.expm1(raw_severity * severity_std + severity_mean) * smearing,
        0.0,
        None,
    )
    result = {
        **dict(setting),
        **metrics(
            targets["occ_validation"],
            targets["cost_validation"],
            probability,
            positive_cost,
        ),
        "best_validation_total_loss": float(
            occurrence_info["best_validation_loss"]
            + LOG_MSE_WEIGHT * severity_info["best_validation_loss"]
        ),
        "best_epoch": int(
            max(occurrence_info["best_epoch"], severity_info["best_epoch"])
        ),
        "parameters": int(
            occurrence_info["parameters"] + severity_info["parameters"]
        ),
        "input_dim": int(matrix["input_dim"]),
        "smearing_factor": float(smearing),
        "status": "success",
    }
    del occurrence_model, severity_model
    gc.collect()
    return result


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    metrics_to_summarize = [
        "best_validation_total_loss",
        "roc_auc",
        "brier",
        "soft_cost_mae",
        "soft_cost_rmse",
        "severity_rmse",
        "portfolio_error_pct",
        "parameters",
        "input_dim",
    ]
    rows: List[Dict[str, Any]] = []
    for (block, setting_id), group in runs.groupby(["ablation_block", "setting_id"]):
        row: Dict[str, Any] = {
            "ablation_block": block,
            "setting_id": setting_id,
            "setting_label": group.iloc[0]["setting_label"],
            "runs": int(len(group)),
        }
        for name in metrics_to_summarize:
            row[f"{name}_mean"] = float(group[name].mean())
            row[f"{name}_std"] = float(group[name].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def paired_differences(runs: pd.DataFrame) -> pd.DataFrame:
    comparisons = {
        "architecture": ("shared_m3_log_mse", "separate_m3_log_mse"),
        "objective": ("gamma_m3", "log_mse_m3"),
        "representation": ("gamma_m4", "gamma_m3"),
    }
    rows: List[Dict[str, Any]] = []
    for block, (candidate, reference) in comparisons.items():
        subset = runs[runs["ablation_block"] == block]
        left = subset[subset["setting_id"] == candidate].set_index("seed")
        right = subset[subset["setting_id"] == reference].set_index("seed")
        for seed in sorted(set(left.index).intersection(right.index)):
            rows.append(
                {
                    "ablation_block": block,
                    "seed": int(seed),
                    "candidate": candidate,
                    "reference": reference,
                    "candidate_minus_reference_soft_rmse": float(
                        left.loc[seed, "soft_cost_rmse"]
                        - right.loc[seed, "soft_cost_rmse"]
                    ),
                    "candidate_minus_reference_soft_mae": float(
                        left.loc[seed, "soft_cost_mae"]
                        - right.loc[seed, "soft_cost_mae"]
                    ),
                    "candidate_minus_reference_roc_auc": float(
                        left.loc[seed, "roc_auc"] - right.loc[seed, "roc_auc"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def settings_for(stage: str, seeds: Sequence[int]) -> List[Dict[str, Any]]:
    settings: List[Dict[str, Any]] = []
    if stage in {"all", "lr"}:
        for learning_rate in LEARNING_RATES:
            for seed in seeds:
                settings.append(
                    {
                        "ablation_block": "learning_rate",
                        "setting_id": f"lr_{learning_rate:g}",
                        "setting_label": f"Initial LR={learning_rate:g}",
                        "feature_block": "M3",
                        "architecture": "shared",
                        "objective": "log_mse",
                        "severity_weight": LOG_MSE_WEIGHT,
                        "learning_rate": learning_rate,
                        "seed": seed,
                    }
                )
    if stage in {"all", "architecture"}:
        for architecture, setting_id, label in [
            ("shared", "shared_m3_log_mse", "Shared MTL"),
            ("separate", "separate_m3_log_mse", "Separate MLPs"),
        ]:
            for seed in seeds:
                settings.append(
                    {
                        "ablation_block": "architecture",
                        "setting_id": setting_id,
                        "setting_label": label,
                        "feature_block": "M3",
                        "architecture": architecture,
                        "objective": "log_mse",
                        "severity_weight": LOG_MSE_WEIGHT,
                        "learning_rate": REFERENCE_LR,
                        "seed": seed,
                    }
                )
    if stage in {"all", "objective"}:
        for objective, weight, setting_id, label in [
            ("log_mse", LOG_MSE_WEIGHT, "log_mse_m3", "Standardized log-MSE"),
            ("gamma", GAMMA_WEIGHT, "gamma_m3", "Gamma mean NLL"),
        ]:
            for seed in seeds:
                settings.append(
                    {
                        "ablation_block": "objective",
                        "setting_id": setting_id,
                        "setting_label": label,
                        "feature_block": "M3",
                        "architecture": "shared",
                        "objective": objective,
                        "severity_weight": weight,
                        "learning_rate": REFERENCE_LR,
                        "seed": seed,
                    }
                )
    if stage in {"all", "representation"}:
        for feature_block, setting_id, label in [
            ("M3", "gamma_m3", "M3 Gamma-MTL"),
            ("M4", "gamma_m4", "M4 Gamma-MTL"),
        ]:
            for seed in seeds:
                settings.append(
                    {
                        "ablation_block": "representation",
                        "setting_id": setting_id,
                        "setting_label": label,
                        "feature_block": feature_block,
                        "architecture": "shared",
                        "objective": "gamma",
                        "severity_weight": GAMMA_WEIGHT,
                        "learning_rate": REFERENCE_LR,
                        "seed": seed,
                    }
                )
    return settings


def main() -> None:
    args = parse_args()
    cfg = PROFILES[args.profile]
    device = configure_cpu()
    project = args.project_dir.expanduser().resolve()
    development_file = (
        project
        / "outputs"
        / "week1_step2"
        / "development_2017_2018_full_exposure.parquet"
    )
    if not development_file.exists():
        raise FileNotFoundError(development_file)
    output_dir = (
        project / "outputs" / "unified_m3_paper_ablations" / args.profile
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    all_runs_path = output_dir / "table_01_all_ablation_runs.csv"
    if all_runs_path.exists() and not args.force:
        existing = pd.read_csv(all_runs_path)
    else:
        existing = pd.DataFrame()

    started = time.time()
    raw = pd.read_parquet(development_file).reset_index(drop=True)
    required = {
        "ID_policy",
        "target_occurrence",
        "target_cost",
        "cost_claims_year",
        "n_medical_services",
        "premium",
        "age",
        "type_product",
    }
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise KeyError(f"Development file missing required columns: {missing}")
    if (pd.to_numeric(raw["target_cost"], errors="raise") < 0).any():
        raise ValueError("target_cost contains negative values.")

    m3 = engineer_m3(raw)
    train_idx, validation_idx = grouped_split(m3)
    m4_context = fit_m4_context(m3.iloc[train_idx])
    m4 = apply_m4(raw, m4_context)
    matrices = {
        "M3": prepare_matrix(m3, train_idx, validation_idx, "M3", output_dir),
        "M4": prepare_matrix(m4, train_idx, validation_idx, "M4", output_dir),
    }
    occurrence = pd.to_numeric(raw["target_occurrence"], errors="raise").to_numpy(
        dtype=np.int64
    )
    cost = pd.to_numeric(raw["target_cost"], errors="raise").to_numpy(
        dtype=np.float64
    )
    targets = {
        "occ_train": occurrence[train_idx],
        "occ_validation": occurrence[validation_idx],
        "cost_train": cost[train_idx],
        "cost_validation": cost[validation_idx],
    }

    rows = existing.to_dict("records") if not existing.empty else []
    completed = {
        (str(row["ablation_block"]), str(row["setting_id"]), int(row["seed"]))
        for row in rows
        if row.get("status") == "success"
    }
    settings = settings_for(args.stage, cfg["seeds"])
    # Identical configurations occur in several logical blocks. Train each
    # unique configuration once and copy its measured result into the other
    # block, preserving the paper's block-wise presentation without wasting
    # CPU time or introducing a second stochastic realization.
    def signature(item: Mapping[str, Any]) -> Tuple[Any, ...]:
        return (
            str(item["feature_block"]),
            str(item["architecture"]),
            str(item["objective"]),
            float(item["severity_weight"]),
            float(item["learning_rate"]),
            int(item["seed"]),
        )

    result_cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for row in rows:
        if row.get("status") == "success":
            result_cache[signature(row)] = dict(row)

    for index, setting in enumerate(settings, start=1):
        key = (
            str(setting["ablation_block"]),
            str(setting["setting_id"]),
            int(setting["seed"]),
        )
        if key in completed and not args.force:
            print(f"[{index}/{len(settings)}] Resume {key}", flush=True)
            continue
        cached = result_cache.get(signature(setting))
        if cached is not None and not args.force:
            result = {
                **cached,
                **dict(setting),
                "status": "success",
                "reused_identical_configuration": True,
            }
            rows.append(result)
            completed.add(key)
            pd.DataFrame(rows).to_csv(all_runs_path, index=False)
            print(f"[{index}/{len(settings)}] Reuse identical configuration {key}", flush=True)
            continue
        print(f"[{index}/{len(settings)}] Run {key}", flush=True)
        try:
            matrix = matrices[str(setting["feature_block"])]
            if setting["architecture"] == "separate":
                result = run_separate_setting(setting, matrix, targets, cfg, device)
            else:
                result = run_shared_setting(setting, matrix, targets, cfg, device)
        except Exception as exc:
            result = {**setting, "status": "failed", "error": repr(exc)}
            pd.DataFrame(rows + [result]).to_csv(all_runs_path, index=False)
            raise
        rows.append(result)
        result_cache[signature(setting)] = dict(result)
        completed.add(key)
        pd.DataFrame(rows).to_csv(all_runs_path, index=False)

    runs = pd.DataFrame(rows)
    selected_keys = {
        (str(s["ablation_block"]), str(s["setting_id"]), int(s["seed"]))
        for s in settings
    }
    runs = runs[
        runs.apply(
            lambda row: (
                str(row["ablation_block"]),
                str(row["setting_id"]),
                int(row["seed"]),
            )
            in selected_keys,
            axis=1,
        )
    ].copy()
    successful = runs[runs["status"] == "success"].copy()
    summary = summarize(successful)
    summary.to_csv(output_dir / "table_02_ablation_summary_mean_std.csv", index=False)
    paired = paired_differences(successful)
    paired.to_csv(output_dir / "table_03_paired_seed_differences.csv", index=False)

    protocol = {
        "study": "Unified development-only M3 paper ablations",
        "development_file": str(development_file),
        "temporal_2018_2019_loaded": False,
        "development_rows": int(len(raw)),
        "training_rows": int(len(train_idx)),
        "validation_rows": int(len(validation_idx)),
        "policy_overlap": 0,
        "split": "GroupShuffleSplit by ID_policy, fixed 80/20",
        "split_seed": SPLIT_SEED,
        "model_seeds": list(cfg["seeds"]),
        "epochs": int(cfg["epochs"]),
        "batch_size": int(cfg["batch_size"]),
        "learning_rates_screened": LEARNING_RATES,
        "reference_learning_rate": REFERENCE_LR,
        "log_mse_weight": LOG_MSE_WEIGHT,
        "gamma_weight": GAMMA_WEIGHT,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip": GRADIENT_CLIP,
        "scheduler": "ReduceLROnPlateau",
        "scheduler_factor": SCHEDULER_FACTOR,
        "scheduler_patience": SCHEDULER_PATIENCE,
        "m3_input_dim": matrices["M3"]["input_dim"],
        "m4_input_dim": matrices["M4"]["input_dim"],
        "m3_features": matrices["M3"]["features"],
        "m4_added_features": M4_ADDED_NUMERIC,
        "primary_endpoint": "validation soft-cost RMSE",
        "soft_cost_definition": "predicted occurrence probability * predicted positive conditional mean",
        "interpretation_boundary": "Comparisons are interpreted only within each controlled ablation block.",
        "elapsed_minutes": float((time.time() - started) / 60.0),
    }
    save_json(protocol, output_dir / "protocol_and_integrity.json")
    save_json(m4_context, output_dir / "m4_train_fitted_context.json")

    print("\nCompleted unified development-only ablation suite.")
    print(summary.to_string(index=False))
    print(f"\nOutputs: {output_dir}")


if __name__ == "__main__":
    main()
