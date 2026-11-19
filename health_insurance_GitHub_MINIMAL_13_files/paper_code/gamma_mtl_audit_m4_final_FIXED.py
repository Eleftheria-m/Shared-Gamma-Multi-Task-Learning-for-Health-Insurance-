#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Final data audit + M4 feature ablation for the Shared Gamma-MTL model.

Purpose
-------
This script answers one final, pre-declared question:

    Is the remaining prediction ceiling caused by inadequate feature
    representation rather than by the Gamma-MTL architecture?

It first audits data quality, temporal drift and the contribution of the
high-cost tail to RMSE. It then compares:

    M3 Shared Gamma-MTL
    versus
    M4 Shared Gamma-MTL

The neural architecture, Gamma objective and training settings remain fixed.
Only the input representation changes.

Compatibility fix
-----------------
Categorical and boolean columns are converted to plain Python strings with
numpy.nan missing values before SimpleImputer. This avoids the sklearn 1.3
"bool object has no attribute transpose" failure caused by pandas nullable
values.

M4 features
-----------
The M4 block contains a small set of actuarially motivated derived features:

1. log previous loss ratio
       log(1 + previous_cost / (premium + 1))

2. log previous cost per service
       log(1 + previous_cost / (previous_services + 1))

3. previous cost intensity
       log(1 + previous_cost) - log(1 + previous_services)

4. age × previous occurrence

5. previous service intensity × previous occurrence

6. product-relative log premium
       log_premium - training median log_premium within product

The product medians, preprocessing statistics, missing-value treatment and
one-hot vocabulary are fitted on the development training subset only.

Scientific boundary
-------------------
The 2018->2019 temporal set has already been inspected in previous project
experiments. Therefore:

- M4 is designed and selected using development data only.
- The script writes a freeze record before temporal evaluation.
- Temporal M4 results must be described as a frozen post-hoc out-of-time
  benchmark, not as a newly untouched confirmatory test.
- By default, temporal M4 evaluation runs only if the development promotion
  rule is met. Use --force-temporal only for an explicitly exploratory result.

Existing M3 result
------------------
The existing Shared Gamma-MTL was already evaluated on the 2018->2019 cohort.
This script reuses its frozen checkpoints for a strictly aligned comparison.

Outputs
-------
outputs/gamma_mtl_audit_m4_final/<profile>/
    audit/
    development/
    temporal/
    paper_tables/
    plots/
    protocol_and_integrity.json
    README_RESULTS.txt

Recommended commands
--------------------
Quick verification:
    python -u gamma_mtl_audit_m4_final.py --profile quick --stage all

Final paper run:
    python -u gamma_mtl_audit_m4_final.py --profile paper --stage all \
        2>&1 | tee gamma_mtl_audit_m4_final.log
"""

# =============================================================================
# Thread limits before numerical imports
# =============================================================================

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

# =============================================================================
# Imports
# =============================================================================

import argparse
import copy
import gc
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_gamma_deviance,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# Fixed protocol
# =============================================================================

DEFAULT_PROJECT_DIR = Path(
    os.environ.get(
        "HEALTH_PROJECT_DIR",
        "/home/eleut/health_insurance_project",
    )
).expanduser()

PROFILES: Dict[str, Dict[str, Any]] = {
    "quick": {
        "seeds": [42],
        "epochs": 3,
        "batch_size": 512,
        "bootstrap_replicates": 50,
    },
    "paper": {
        "seeds": [42, 123, 2026, 31415],
        "epochs": 35,
        "batch_size": 512,
        "bootstrap_replicates": 500,
    },
}

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GAMMA_WEIGHT = 0.10
GRADIENT_CLIP = 5.0
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 3

VALIDATION_SIZE = 0.20
SPLIT_RANDOM_STATE = 42

MIN_LOG_MEAN = -8.0
MAX_LOG_MEAN = 8.0
EPS = 1e-8

PRIMARY_METRIC = "soft_cost_rmse"
ROC_GUARDRAIL_TOLERANCE = 0.002

M4_FEATURE_NAMES = [
    "m4_log_prev_loss_ratio",
    "m4_log_prev_cost_per_service",
    "m4_cost_intensity",
    "m4_age_prev_occurrence",
    "m4_services_prev_occurrence",
    "m4_relative_log_premium_product",
]


# =============================================================================
# General utilities
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Shared Gamma-MTL data audit and M4 feature ablation."
    )

    parser.add_argument(
        "--project-dir",
        type=Path,
        default=DEFAULT_PROJECT_DIR,
    )

    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="paper",
    )

    parser.add_argument(
        "--stage",
        choices=[
            "all",
            "audit",
            "development",
            "temporal",
        ],
        default="all",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute completed M4 seed folders.",
    )

    parser.add_argument(
        "--force-temporal",
        action="store_true",
        help=(
            "Run temporal M4 evaluation even when the development "
            "promotion rule is not met. The result is exploratory."
        ),
    )

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


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)

    raise TypeError(
        "Not JSON serializable: {}".format(type(value).__name__)
    )


def save_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            default=json_default,
        ),
        encoding="utf-8",
    )


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def safe_torch_load(
    path: Path,
    device: torch.device,
) -> Mapping[str, Any]:
    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=device,
        )


def dense_array(
    values: Any,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    if sp.issparse(values):
        values = values.toarray()

    result = np.asarray(values, dtype=dtype)

    if result.ndim != 2:
        raise ValueError(
            "Expected a 2D matrix, got shape {}.".format(result.shape)
        )

    if not np.isfinite(result).all():
        raise ValueError("Encoded matrix contains non-finite values.")

    return result


def rmse(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    return float(
        np.sqrt(
            mean_squared_error(actual, predicted)
        )
    )


def portfolio_error_pct(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    actual_total = float(np.sum(actual))

    if actual_total == 0:
        return np.nan

    return float(
        100.0
        * (
            float(np.sum(predicted))
            - actual_total
        )
        / actual_total
    )


def count_parameters(model: nn.Module) -> int:
    return int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
    )


# =============================================================================
# Paths
# =============================================================================

def resolve_paths(
    project_dir: Path,
    profile: str,
) -> Dict[str, Path]:
    project_dir = project_dir.expanduser().resolve()

    paths = {
        "project_dir": project_dir,
        "development_file": (
            project_dir
            / "outputs"
            / "week1_step2"
            / "development_2017_2018_full_exposure.parquet"
        ),
        "temporal_file": (
            project_dir
            / "outputs"
            / "week1_step2"
            / "temporal_test_2018_2019_full_exposure.parquet"
        ),
        "m3_preprocessor": (
            project_dir
            / "outputs"
            / "final_mse_ablations_35epochs_4runs"
            / "preprocessors"
            / "M3_basic_history_context_preprocessor.joblib"
        ),
        "m3_checkpoint_root": (
            project_dir
            / "outputs"
            / "gamma_mtl_objective_ablation"
            / "paper"
            / "runs"
            / "gamma"
            / "lambda_0p10"
        ),
        "output_root": (
            project_dir
            / "outputs"
            / "gamma_mtl_audit_m4_final"
            / profile
        ),
    }

    required = [
        paths["development_file"],
        paths["temporal_file"],
        paths["m3_preprocessor"],
    ]

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required files:\n{}".format(
                "\n".join(str(path) for path in missing)
            )
        )

    return paths


# =============================================================================
# Feature engineering
# =============================================================================

def engineer_m3_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    result = data.copy()

    previous_cost = pd.to_numeric(
        result["cost_claims_year"],
        errors="coerce",
    )

    previous_services = pd.to_numeric(
        result["n_medical_services"],
        errors="coerce",
    )

    premium = pd.to_numeric(
        result["premium"],
        errors="coerce",
    )

    result["prev_occurrence"] = (
        previous_cost > 0
    ).astype("float32")

    result["log_prev_cost"] = np.log1p(
        previous_cost.clip(lower=0)
    ).astype("float32")

    result["log_prev_services"] = np.log1p(
        previous_services.clip(lower=0)
    ).astype("float32")

    result["log_premium"] = np.log1p(
        premium.clip(lower=0)
    ).astype("float32")

    return result


def fit_m4_context(
    development_train: pd.DataFrame,
) -> Dict[str, Any]:
    log_premium = pd.to_numeric(
        development_train["log_premium"],
        errors="coerce",
    )

    global_median = float(
        log_premium.median()
    )

    product_column = (
        "type_product"
        if "type_product" in development_train.columns
        else None
    )

    product_medians: Dict[str, float] = {}

    if product_column is not None:
        products = (
            development_train[product_column]
            .astype("string")
            .fillna("Missing")
        )

        temporary = pd.DataFrame(
            {
                "product": products,
                "log_premium": log_premium,
            }
        )

        medians = temporary.groupby(
            "product",
            dropna=False,
        )["log_premium"].median()

        product_medians = {
            str(key): float(value)
            for key, value in medians.items()
            if pd.notna(value)
        }

    return {
        "product_column": product_column,
        "global_log_premium_median": global_median,
        "product_log_premium_medians": product_medians,
    }


def apply_m4_features(
    data: pd.DataFrame,
    context: Mapping[str, Any],
) -> pd.DataFrame:
    result = engineer_m3_features(data)

    previous_cost = pd.to_numeric(
        result["cost_claims_year"],
        errors="coerce",
    ).clip(lower=0)

    previous_services = pd.to_numeric(
        result["n_medical_services"],
        errors="coerce",
    ).clip(lower=0)

    premium = pd.to_numeric(
        result["premium"],
        errors="coerce",
    ).clip(lower=0)

    result["m4_log_prev_loss_ratio"] = np.log1p(
        previous_cost
        / (premium + 1.0)
    ).astype("float32")

    result["m4_log_prev_cost_per_service"] = np.log1p(
        previous_cost
        / (previous_services + 1.0)
    ).astype("float32")

    result["m4_cost_intensity"] = (
        result["log_prev_cost"]
        - result["log_prev_services"]
    ).astype("float32")

    if "age" in result.columns:
        age = pd.to_numeric(
            result["age"],
            errors="coerce",
        )

        result["m4_age_prev_occurrence"] = (
            age
            * result["prev_occurrence"]
        ).astype("float32")
    else:
        result["m4_age_prev_occurrence"] = np.nan

    result["m4_services_prev_occurrence"] = (
        result["log_prev_services"]
        * result["prev_occurrence"]
    ).astype("float32")

    product_column = context.get(
        "product_column"
    )

    global_median = float(
        context[
            "global_log_premium_median"
        ]
    )

    product_medians = {
        str(key): float(value)
        for key, value in context.get(
            "product_log_premium_medians",
            {},
        ).items()
    }

    if (
        product_column is not None
        and product_column in result.columns
    ):
        product_keys = (
            result[product_column]
            .astype("string")
            .fillna("Missing")
        )

        reference = product_keys.map(
            product_medians
        ).astype(float)

        reference = reference.fillna(
            global_median
        )
    else:
        reference = pd.Series(
            global_median,
            index=result.index,
            dtype=float,
        )

    result["m4_relative_log_premium_product"] = (
        pd.to_numeric(
            result["log_premium"],
            errors="coerce",
        )
        - reference
    ).astype("float32")

    return result


def infer_feature_types(
    train_frame: pd.DataFrame,
    feature_names: Sequence[str],
) -> Tuple[List[str], List[str]]:
    """
    Split raw features into numeric and categorical columns.

    Boolean columns are treated as categorical. This avoids the pandas
    nullable-boolean / SimpleImputer incompatibility observed with
    scikit-learn 1.3 and Python 3.8.
    """
    numeric_columns: List[str] = []
    categorical_columns: List[str] = []

    for column in feature_names:
        series = train_frame[column]

        if pd.api.types.is_bool_dtype(series.dtype):
            categorical_columns.append(column)
        elif pd.api.types.is_numeric_dtype(series.dtype):
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)

    return (
        numeric_columns,
        categorical_columns,
    )


def sanitize_sklearn_frame(
    frame: pd.DataFrame,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> pd.DataFrame:
    """
    Convert every column to a representation that is safe for sklearn 1.3.

    Important:
    - nullable booleans are converted to categorical strings;
    - pandas.NA is replaced by numpy.nan;
    - numeric infinities are converted to numpy.nan;
    - categorical columns contain only Python strings or numpy.nan.

    Using pandas StringDtype with pd.NA can make SimpleImputer 1.3 create a
    scalar boolean mask and fail with:
        AttributeError: 'bool' object has no attribute 'transpose'
    """
    result = frame.copy()

    for column in numeric_columns:
        numeric = pd.to_numeric(
            result[column],
            errors="coerce",
        )

        numeric = numeric.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        result[column] = numeric.astype(
            np.float64
        )

    for column in categorical_columns:
        values = result[column]

        converted = values.map(
            lambda value: (
                np.nan
                if pd.isna(value)
                else str(value)
            )
        )

        result[column] = converted.astype(
            object
        )

    return result


def build_m4_preprocessor(
    train_frame: pd.DataFrame,
    feature_names: Sequence[str],
) -> Tuple[
    ColumnTransformer,
    List[str],
    List[str],
]:
    (
        numeric_columns,
        categorical_columns,
    ) = infer_feature_types(
        train_frame,
        feature_names,
    )

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="Missing",
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=np.float32,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=True,
    )

    clean_train = sanitize_sklearn_frame(
        train_frame[list(feature_names)],
        numeric_columns,
        categorical_columns,
    )

    preprocessor.fit(clean_train)

    return (
        preprocessor,
        numeric_columns,
        categorical_columns,
    )


def transform_m4(
    preprocessor: ColumnTransformer,
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    categorical_columns: Sequence[str],
) -> np.ndarray:
    numeric_columns = [
        column
        for column in feature_names
        if column not in set(
            categorical_columns
        )
    ]

    clean = sanitize_sklearn_frame(
        frame[list(feature_names)],
        numeric_columns,
        categorical_columns,
    )

    return dense_array(
        preprocessor.transform(clean),
        dtype=np.float32,
    )


# =============================================================================
# Grouped split
# =============================================================================

def grouped_split(
    development: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_SIZE,
        random_state=SPLIT_RANDOM_STATE,
    )

    train_positions, validation_positions = next(
        splitter.split(
            development,
            groups=development["ID_policy"],
        )
    )

    train_ids = set(
        development.iloc[
            train_positions
        ]["ID_policy"].astype(str)
    )

    validation_ids = set(
        development.iloc[
            validation_positions
        ]["ID_policy"].astype(str)
    )

    if train_ids & validation_ids:
        raise RuntimeError(
            "Policy leakage detected in development split."
        )

    return (
        np.asarray(train_positions, dtype=int),
        np.asarray(validation_positions, dtype=int),
    )


# =============================================================================
# Frozen Shared Gamma-MTL architecture
# =============================================================================

class SharedGammaMTL(nn.Module):
    def __init__(
        self,
        input_dim: int,
    ) -> None:
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
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.severity_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        shared = self.shared(features)

        occurrence_logit = (
            self.occurrence_head(shared)
            .squeeze(1)
        )

        raw_severity = (
            self.severity_head(shared)
            .squeeze(1)
        )

        return (
            occurrence_logit,
            raw_severity,
        )


def gamma_mean_nll(
    raw_log_mean: torch.Tensor,
    target_scaled: torch.Tensor,
) -> torch.Tensor:
    log_mean = torch.clamp(
        raw_log_mean,
        min=MIN_LOG_MEAN,
        max=MAX_LOG_MEAN,
    )

    mean = torch.exp(log_mean)

    return torch.mean(
        target_scaled
        / torch.clamp(
            mean,
            min=EPS,
        )
        + log_mean
    )


# =============================================================================
# Training and prediction
# =============================================================================

def create_loader(
    features: np.ndarray,
    occurrence: np.ndarray,
    cost: np.ndarray,
    gamma_scale: float,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
) -> DataLoader:
    scaled_cost = np.zeros(
        len(cost),
        dtype=np.float32,
    )

    positive = occurrence == 1

    scaled_cost[positive] = (
        cost[positive]
        / gamma_scale
    ).astype(np.float32)

    dataset = TensorDataset(
        torch.from_numpy(
            features.astype(
                np.float32,
                copy=False,
            )
        ),
        torch.from_numpy(
            occurrence.astype(
                np.float32,
                copy=False,
            )
        ),
        torch.from_numpy(
            scaled_cost
        ),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,
        pin_memory=False,
    )


def evaluate_objective(
    model: SharedGammaMTL,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()

    bce = nn.BCEWithLogitsLoss()

    rows_total = 0
    positives_total = 0

    occurrence_sum = 0.0
    severity_sum = 0.0

    with torch.no_grad():
        for (
            features,
            occurrence,
            scaled_cost,
        ) in loader:
            features = features.to(device)
            occurrence = occurrence.to(device)
            scaled_cost = scaled_cost.to(device)

            occurrence_logit, raw_severity = model(
                features
            )

            occurrence_loss = bce(
                occurrence_logit,
                occurrence,
            )

            positive_mask = occurrence > 0.5
            positive_count = int(
                positive_mask.sum().item()
            )

            if positive_count:
                severity_loss = gamma_mean_nll(
                    raw_severity[
                        positive_mask
                    ],
                    scaled_cost[
                        positive_mask
                    ],
                )
            else:
                severity_loss = (
                    raw_severity.sum()
                    * 0.0
                )

            rows = len(features)

            occurrence_sum += (
                float(
                    occurrence_loss.item()
                )
                * rows
            )

            if positive_count:
                severity_sum += (
                    float(
                        severity_loss.item()
                    )
                    * positive_count
                )

            rows_total += rows
            positives_total += positive_count

    occurrence_mean = (
        occurrence_sum
        / rows_total
    )

    severity_mean = (
        severity_sum
        / positives_total
        if positives_total
        else 0.0
    )

    return {
        "total_loss": float(
            occurrence_mean
            + GAMMA_WEIGHT
            * severity_mean
        ),
        "occurrence_loss": float(
            occurrence_mean
        ),
        "severity_loss": float(
            severity_mean
        ),
    }


def train_m4_seed(
    seed: int,
    X_train: np.ndarray,
    y_occ_train: np.ndarray,
    y_cost_train: np.ndarray,
    X_validation: np.ndarray,
    y_occ_validation: np.ndarray,
    y_cost_validation: np.ndarray,
    gamma_scale: float,
    profile_cfg: Mapping[str, Any],
    device: torch.device,
    run_dir: Path,
) -> Tuple[
    SharedGammaMTL,
    Dict[str, Any],
]:
    set_seed(seed)

    train_loader = create_loader(
        X_train,
        y_occ_train,
        y_cost_train,
        gamma_scale,
        batch_size=int(
            profile_cfg[
                "batch_size"
            ]
        ),
        shuffle=True,
        drop_last=True,
    )

    validation_loader = create_loader(
        X_validation,
        y_occ_validation,
        y_cost_validation,
        gamma_scale,
        batch_size=1024,
        shuffle=False,
        drop_last=False,
    )

    model = SharedGammaMTL(
        input_dim=X_train.shape[1]
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=SCHEDULER_FACTOR,
            patience=SCHEDULER_PATIENCE,
        )
    )

    bce = nn.BCEWithLogitsLoss()

    best_loss = math.inf
    best_epoch = -1
    best_state = None
    history: List[
        Dict[str, Any]
    ] = []

    epochs = int(
        profile_cfg["epochs"]
    )

    for epoch in range(
        1,
        epochs + 1,
    ):
        model.train()

        rows_total = 0
        positives_total = 0

        occurrence_sum = 0.0
        severity_sum = 0.0

        for (
            features,
            occurrence,
            scaled_cost,
        ) in train_loader:
            features = features.to(device)
            occurrence = occurrence.to(device)
            scaled_cost = scaled_cost.to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            occurrence_logit, raw_severity = model(
                features
            )

            occurrence_loss = bce(
                occurrence_logit,
                occurrence,
            )

            positive_mask = occurrence > 0.5
            positive_count = int(
                positive_mask.sum().item()
            )

            if positive_count:
                severity_loss = gamma_mean_nll(
                    raw_severity[
                        positive_mask
                    ],
                    scaled_cost[
                        positive_mask
                    ],
                )
            else:
                severity_loss = (
                    raw_severity.sum()
                    * 0.0
                )

            total_loss = (
                occurrence_loss
                + GAMMA_WEIGHT
                * severity_loss
            )

            if not torch.isfinite(
                total_loss
            ):
                raise FloatingPointError(
                    "Non-finite loss at seed {} epoch {}.".format(
                        seed,
                        epoch,
                    )
                )

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRADIENT_CLIP,
            )

            optimizer.step()

            rows = len(features)

            occurrence_sum += (
                float(
                    occurrence_loss.item()
                )
                * rows
            )

            if positive_count:
                severity_sum += (
                    float(
                        severity_loss.item()
                    )
                    * positive_count
                )

            rows_total += rows
            positives_total += positive_count

        train_occurrence = (
            occurrence_sum
            / rows_total
        )

        train_severity = (
            severity_sum
            / positives_total
            if positives_total
            else 0.0
        )

        train_total = (
            train_occurrence
            + GAMMA_WEIGHT
            * train_severity
        )

        validation_objective = evaluate_objective(
            model,
            validation_loader,
            device,
        )

        scheduler.step(
            validation_objective[
                "total_loss"
            ]
        )

        current_lr = float(
            optimizer
            .param_groups[0]["lr"]
        )

        history.append(
            {
                "epoch": epoch,
                "train_total_loss": train_total,
                "train_occurrence_loss": (
                    train_occurrence
                ),
                "train_severity_loss": (
                    train_severity
                ),
                "validation_total_loss": (
                    validation_objective[
                        "total_loss"
                    ]
                ),
                "validation_occurrence_loss": (
                    validation_objective[
                        "occurrence_loss"
                    ]
                ),
                "validation_severity_loss": (
                    validation_objective[
                        "severity_loss"
                    ]
                ),
                "learning_rate": current_lr,
            }
        )

        if (
            validation_objective[
                "total_loss"
            ]
            < best_loss
        ):
            best_loss = float(
                validation_objective[
                    "total_loss"
                ]
            )

            best_epoch = int(epoch)

            best_state = copy.deepcopy(
                model.state_dict()
            )

        print(
            "M4 | seed={} | epoch={:02d}/{:02d} | "
            "train={:.6f} | val={:.6f} | lr={:.2e}".format(
                seed,
                epoch,
                epochs,
                train_total,
                validation_objective[
                    "total_loss"
                ],
                current_lr,
            ),
            flush=True,
        )

    if best_state is None:
        raise RuntimeError(
            "No M4 checkpoint captured."
        )

    model.load_state_dict(
        best_state
    )

    checkpoint = {
        "model_type": (
            "SharedGammaMTL"
        ),
        "feature_block": "M4",
        "seed": int(seed),
        "input_dim": int(
            X_train.shape[1]
        ),
        "gamma_scale": float(
            gamma_scale
        ),
        "gamma_weight": float(
            GAMMA_WEIGHT
        ),
        "learning_rate": float(
            LEARNING_RATE
        ),
        "weight_decay": float(
            WEIGHT_DECAY
        ),
        "best_epoch": int(
            best_epoch
        ),
        "best_validation_loss": float(
            best_loss
        ),
        "parameter_count": int(
            count_parameters(
                model
            )
        ),
        "model_state_dict": (
            best_state
        ),
    }

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        checkpoint,
        run_dir
        / "best_model.pt",
    )

    pd.DataFrame(
        history
    ).to_csv(
        run_dir
        / "training_history.csv",
        index=False,
    )

    return (
        model,
        checkpoint,
    )


def load_model(
    checkpoint_file: Path,
    device: torch.device,
) -> Tuple[
    SharedGammaMTL,
    Mapping[str, Any],
]:
    checkpoint = safe_torch_load(
        checkpoint_file,
        device,
    )

    model = SharedGammaMTL(
        input_dim=int(
            checkpoint["input_dim"]
        )
    ).to(device)

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    return (
        model,
        checkpoint,
    )


def predict_model(
    model: SharedGammaMTL,
    checkpoint: Mapping[str, Any],
    features: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> Dict[str, np.ndarray]:
    scale = float(
        checkpoint.get(
            "gamma_scale",
            checkpoint.get(
                "gamma_cost_scale"
            ),
        )
    )

    probabilities: List[
        np.ndarray
    ] = []

    severities: List[
        np.ndarray
    ] = []

    model.eval()

    with torch.no_grad():
        for start in range(
            0,
            len(features),
            batch_size,
        ):
            stop = min(
                start + batch_size,
                len(features),
            )

            batch = (
                torch.from_numpy(
                    features[
                        start:stop
                    ].astype(
                        np.float32,
                        copy=False,
                    )
                )
                .to(device)
            )

            occurrence_logit, raw_severity = model(
                batch
            )

            probability = torch.sigmoid(
                occurrence_logit
            )

            severity = (
                torch.exp(
                    torch.clamp(
                        raw_severity,
                        min=MIN_LOG_MEAN,
                        max=MAX_LOG_MEAN,
                    )
                )
                * scale
            )

            probabilities.append(
                probability
                .cpu()
                .numpy()
            )

            severities.append(
                severity
                .cpu()
                .numpy()
            )

    probability = np.concatenate(
        probabilities
    )

    positive_severity = np.concatenate(
        severities
    )

    return {
        "probability": probability,
        "positive_severity": (
            positive_severity
        ),
        "soft_cost": (
            probability
            * positive_severity
        ),
    }


# =============================================================================
# Metrics
# =============================================================================

def calculate_metrics(
    actual_occurrence: np.ndarray,
    actual_cost: np.ndarray,
    predictions: Mapping[str, np.ndarray],
) -> Dict[str, float]:
    probability = np.asarray(
        predictions["probability"],
        dtype=float,
    )

    positive_severity = np.asarray(
        predictions[
            "positive_severity"
        ],
        dtype=float,
    )

    soft_cost = np.asarray(
        predictions["soft_cost"],
        dtype=float,
    )

    positive_mask = (
        actual_occurrence == 1
    )

    actual_positive = (
        actual_cost[
            positive_mask
        ]
    )

    predicted_positive = np.clip(
        positive_severity[
            positive_mask
        ],
        EPS,
        None,
    )

    portfolio_error = (
        portfolio_error_pct(
            actual_cost,
            soft_cost,
        )
    )

    return {
        "roc_auc": float(
            roc_auc_score(
                actual_occurrence,
                probability,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                actual_occurrence,
                probability,
            )
        ),
        "brier_score": float(
            brier_score_loss(
                actual_occurrence,
                probability,
            )
        ),
        "soft_cost_mae": float(
            mean_absolute_error(
                actual_cost,
                soft_cost,
            )
        ),
        "soft_cost_rmse": rmse(
            actual_cost,
            soft_cost,
        ),
        "soft_portfolio_error_pct": float(
            portfolio_error
        ),
        "soft_abs_portfolio_error_pct": float(
            abs(
                portfolio_error
            )
        ),
        "positive_severity_mae": float(
            mean_absolute_error(
                actual_positive,
                predicted_positive,
            )
        ),
        "positive_severity_rmse": rmse(
            actual_positive,
            predicted_positive,
        ),
        "positive_severity_gamma_deviance": float(
            mean_gamma_deviance(
                actual_positive,
                predicted_positive,
            )
        ),
    }


def prediction_frame(
    metadata: pd.DataFrame,
    actual_occurrence: np.ndarray,
    actual_cost: np.ndarray,
    predictions: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": np.arange(
                len(actual_cost),
                dtype=int,
            ),
            "ID_policy": (
                metadata["ID_policy"]
                .astype(str)
                .to_numpy()
            ),
            "target_occurrence": (
                actual_occurrence
            ),
            "target_cost": (
                actual_cost
            ),
            "predicted_probability": (
                predictions[
                    "probability"
                ]
            ),
            "predicted_positive_cost": (
                predictions[
                    "positive_severity"
                ]
            ),
            "predicted_soft_cost": (
                predictions[
                    "soft_cost"
                ]
            ),
        }
    )


def ensemble_frames(
    frames: Sequence[
        pd.DataFrame
    ],
) -> pd.DataFrame:
    if not frames:
        raise ValueError(
            "No prediction frames."
        )

    keys = [
        "row_id",
        "ID_policy",
        "target_occurrence",
        "target_cost",
    ]

    result = frames[0][keys].copy()

    for column in [
        "predicted_probability",
        "predicted_positive_cost",
        "predicted_soft_cost",
    ]:
        matrix = np.column_stack(
            [
                frame[column].to_numpy(
                    dtype=float
                )
                for frame in frames
            ]
        )

        result[column] = matrix.mean(
            axis=1
        )

    return result


# =============================================================================
# Audit
# =============================================================================

def dataset_summary(
    name: str,
    frame: pd.DataFrame,
) -> Dict[str, Any]:
    occurrence = pd.to_numeric(
        frame["target_occurrence"],
        errors="coerce",
    )

    cost = pd.to_numeric(
        frame["target_cost"],
        errors="coerce",
    )

    positive = cost > 0
    positive_cost = cost[positive]

    return {
        "cohort": name,
        "rows": int(len(frame)),
        "unique_policies": int(
            frame["ID_policy"]
            .astype(str)
            .nunique()
        ),
        "duplicate_policy_rows": int(
            frame["ID_policy"]
            .astype(str)
            .duplicated()
            .sum()
        ),
        "occurrence_rate": float(
            occurrence.mean()
        ),
        "cost_mean": float(
            cost.mean()
        ),
        "cost_median": float(
            cost.median()
        ),
        "cost_q90": float(
            cost.quantile(0.90)
        ),
        "cost_q95": float(
            cost.quantile(0.95)
        ),
        "cost_q99": float(
            cost.quantile(0.99)
        ),
        "cost_max": float(
            cost.max()
        ),
        "positive_cost_mean": float(
            positive_cost.mean()
        ),
        "positive_cost_median": float(
            positive_cost.median()
        ),
        "positive_cost_q90": float(
            positive_cost.quantile(
                0.90
            )
        ),
        "positive_cost_q95": float(
            positive_cost.quantile(
                0.95
            )
        ),
        "positive_cost_q99": float(
            positive_cost.quantile(
                0.99
            )
        ),
    }


def psi_numeric(
    reference: pd.Series,
    comparison: pd.Series,
    bins: int = 10,
) -> float:
    reference_numeric = pd.to_numeric(
        reference,
        errors="coerce",
    )

    comparison_numeric = pd.to_numeric(
        comparison,
        errors="coerce",
    )

    reference_valid = (
        reference_numeric.dropna()
    )

    if reference_valid.empty:
        return np.nan

    quantiles = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    edges = np.unique(
        reference_valid.quantile(
            quantiles
        ).to_numpy(
            dtype=float
        )
    )

    if len(edges) < 2:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    reference_bins = pd.cut(
        reference_numeric,
        bins=edges,
        include_lowest=True,
    )

    comparison_bins = pd.cut(
        comparison_numeric,
        bins=edges,
        include_lowest=True,
    )

    categories = (
        reference_bins.cat.categories
    )

    reference_counts = (
        reference_bins.value_counts(
            sort=False
        )
        .reindex(
            categories,
            fill_value=0,
        )
        .to_numpy(
            dtype=float
        )
    )

    comparison_counts = (
        comparison_bins.value_counts(
            sort=False
        )
        .reindex(
            categories,
            fill_value=0,
        )
        .to_numpy(
            dtype=float
        )
    )

    reference_counts = np.append(
        reference_counts,
        reference_numeric.isna().sum(),
    )

    comparison_counts = np.append(
        comparison_counts,
        comparison_numeric.isna().sum(),
    )

    reference_pct = (
        reference_counts
        / max(
            reference_counts.sum(),
            1.0,
        )
    )

    comparison_pct = (
        comparison_counts
        / max(
            comparison_counts.sum(),
            1.0,
        )
    )

    reference_pct = np.clip(
        reference_pct,
        1e-6,
        None,
    )

    comparison_pct = np.clip(
        comparison_pct,
        1e-6,
        None,
    )

    return float(
        np.sum(
            (
                comparison_pct
                - reference_pct
            )
            * np.log(
                comparison_pct
                / reference_pct
            )
        )
    )


def standardized_mean_difference(
    reference: pd.Series,
    comparison: pd.Series,
) -> float:
    reference_numeric = pd.to_numeric(
        reference,
        errors="coerce",
    )

    comparison_numeric = pd.to_numeric(
        comparison,
        errors="coerce",
    )

    mean_reference = float(
        reference_numeric.mean()
    )

    mean_comparison = float(
        comparison_numeric.mean()
    )

    variance_reference = float(
        reference_numeric.var(ddof=1)
    )

    variance_comparison = float(
        comparison_numeric.var(ddof=1)
    )

    pooled = math.sqrt(
        max(
            (
                variance_reference
                + variance_comparison
            )
            / 2.0,
            0.0,
        )
    )

    if pooled == 0:
        return 0.0

    return float(
        (
            mean_comparison
            - mean_reference
        )
        / pooled
    )


def categorical_drift(
    reference: pd.Series,
    comparison: pd.Series,
) -> Dict[str, float]:
    reference_values = (
        reference.astype("string")
        .fillna("Missing")
    )

    comparison_values = (
        comparison.astype("string")
        .fillna("Missing")
    )

    reference_distribution = (
        reference_values.value_counts(
            normalize=True
        )
    )

    comparison_distribution = (
        comparison_values.value_counts(
            normalize=True
        )
    )

    categories = sorted(
        set(
            reference_distribution.index
        )
        | set(
            comparison_distribution.index
        )
    )

    reference_vector = np.asarray(
        [
            float(
                reference_distribution.get(
                    category,
                    0.0,
                )
            )
            for category in categories
        ]
    )

    comparison_vector = np.asarray(
        [
            float(
                comparison_distribution.get(
                    category,
                    0.0,
                )
            )
            for category in categories
        ]
    )

    total_variation = float(
        0.5
        * np.sum(
            np.abs(
                comparison_vector
                - reference_vector
            )
        )
    )

    reference_categories = set(
        reference_distribution.index
    )

    unseen_rate = float(
        (~comparison_values.isin(
            reference_categories
        )).mean()
    )

    return {
        "total_variation_distance": (
            total_variation
        ),
        "unseen_category_rate": (
            unseen_rate
        ),
        "reference_categories": int(
            len(
                reference_distribution
            )
        ),
        "comparison_categories": int(
            len(
                comparison_distribution
            )
        ),
    }


def tail_error_table(
    cohort_name: str,
    actual_cost: np.ndarray,
    predicted_cost: np.ndarray,
) -> pd.DataFrame:
    squared_error = (
        actual_cost
        - predicted_cost
    ) ** 2

    total_sse = float(
        squared_error.sum()
    )

    total_cost = float(
        actual_cost.sum()
    )

    rows = []

    for quantile in [
        0.90,
        0.95,
        0.99,
    ]:
        threshold = float(
            np.quantile(
                actual_cost,
                quantile,
            )
        )

        mask = (
            actual_cost >= threshold
        )

        rows.append(
            {
                "cohort": cohort_name,
                "tail": "top_{}pct".format(
                    int(
                        round(
                            100
                            * (
                                1.0
                                - quantile
                            )
                        )
                    )
                ),
                "cost_threshold": threshold,
                "rows": int(
                    mask.sum()
                ),
                "row_share_pct": float(
                    100.0
                    * mask.mean()
                ),
                "actual_cost_share_pct": float(
                    100.0
                    * actual_cost[
                        mask
                    ].sum()
                    / total_cost
                    if total_cost
                    else np.nan
                ),
                "squared_error_share_pct": float(
                    100.0
                    * squared_error[
                        mask
                    ].sum()
                    / total_sse
                    if total_sse
                    else np.nan
                ),
                "segment_rmse": float(
                    np.sqrt(
                        squared_error[
                            mask
                        ].mean()
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Prepare all data
# =============================================================================

def prepare_data(
    paths: Mapping[str, Path],
) -> Dict[str, Any]:
    development_raw = pd.read_parquet(
        paths["development_file"]
    ).reset_index(drop=True)

    temporal_raw = pd.read_parquet(
        paths["temporal_file"]
    ).reset_index(drop=True)

    development_m3 = engineer_m3_features(
        development_raw
    )

    temporal_m3 = engineer_m3_features(
        temporal_raw
    )

    (
        train_positions,
        validation_positions,
    ) = grouped_split(
        development_m3
    )

    m3_preprocessor = joblib.load(
        paths["m3_preprocessor"]
    )

    m3_features = [
        str(name)
        for name in m3_preprocessor.feature_names_in_
    ]

    missing_m3 = [
        column
        for column in m3_features
        if column not in development_m3.columns
    ]

    if missing_m3:
        raise KeyError(
            "Missing M3 features: {}".format(
                missing_m3
            )
        )

    X_m3_development = dense_array(
        m3_preprocessor.transform(
            development_m3[
                m3_features
            ]
        ),
        dtype=np.float32,
    )

    X_m3_temporal = dense_array(
        m3_preprocessor.transform(
            temporal_m3[
                m3_features
            ]
        ),
        dtype=np.float32,
    )

    development_train_raw = (
        development_raw.iloc[
            train_positions
        ].copy()
    )

    m4_context = fit_m4_context(
        engineer_m3_features(
            development_train_raw
        )
    )

    development_m4 = apply_m4_features(
        development_raw,
        m4_context,
    )

    temporal_m4 = apply_m4_features(
        temporal_raw,
        m4_context,
    )

    m4_features = list(
        dict.fromkeys(
            m3_features
            + M4_FEATURE_NAMES
        )
    )

    missing_m4 = [
        column
        for column in m4_features
        if column not in development_m4.columns
    ]

    if missing_m4:
        raise KeyError(
            "Missing M4 features: {}".format(
                missing_m4
            )
        )

    (
        m4_preprocessor,
        numeric_columns,
        categorical_columns,
    ) = build_m4_preprocessor(
        development_m4.iloc[
            train_positions
        ],
        m4_features,
    )

    X_m4_development = transform_m4(
        m4_preprocessor,
        development_m4,
        m4_features,
        categorical_columns,
    )

    X_m4_temporal = transform_m4(
        m4_preprocessor,
        temporal_m4,
        m4_features,
        categorical_columns,
    )

    y_occ_development = pd.to_numeric(
        development_m4[
            "target_occurrence"
        ],
        errors="raise",
    ).to_numpy(
        dtype=np.int64
    )

    y_cost_development = pd.to_numeric(
        development_m4[
            "target_cost"
        ],
        errors="raise",
    ).to_numpy(
        dtype=np.float64
    )

    y_occ_temporal = pd.to_numeric(
        temporal_m4[
            "target_occurrence"
        ],
        errors="raise",
    ).to_numpy(
        dtype=np.int64
    )

    y_cost_temporal = pd.to_numeric(
        temporal_m4[
            "target_cost"
        ],
        errors="raise",
    ).to_numpy(
        dtype=np.float64
    )

    positive_train = (
        y_occ_development[
            train_positions
        ]
        == 1
    )

    gamma_scale = float(
        y_cost_development[
            train_positions
        ][positive_train].mean()
    )

    return {
        "development_raw": (
            development_raw
        ),
        "temporal_raw": (
            temporal_raw
        ),
        "development_m3": (
            development_m3
        ),
        "temporal_m3": (
            temporal_m3
        ),
        "development_m4": (
            development_m4
        ),
        "temporal_m4": (
            temporal_m4
        ),
        "train_positions": (
            train_positions
        ),
        "validation_positions": (
            validation_positions
        ),
        "m3_preprocessor": (
            m3_preprocessor
        ),
        "m4_preprocessor": (
            m4_preprocessor
        ),
        "m3_features": (
            m3_features
        ),
        "m4_features": (
            m4_features
        ),
        "m4_context": (
            m4_context
        ),
        "numeric_columns": (
            numeric_columns
        ),
        "categorical_columns": (
            categorical_columns
        ),
        "X_m3_development": (
            X_m3_development
        ),
        "X_m3_temporal": (
            X_m3_temporal
        ),
        "X_m4_development": (
            X_m4_development
        ),
        "X_m4_temporal": (
            X_m4_temporal
        ),
        "y_occ_development": (
            y_occ_development
        ),
        "y_cost_development": (
            y_cost_development
        ),
        "y_occ_temporal": (
            y_occ_temporal
        ),
        "y_cost_temporal": (
            y_cost_temporal
        ),
        "gamma_scale": (
            gamma_scale
        ),
    }


# =============================================================================
# Frozen M3 predictions
# =============================================================================

def m3_checkpoint_file(
    paths: Mapping[str, Path],
    seed: int,
) -> Path:
    return (
        paths["m3_checkpoint_root"]
        / "seed_{}".format(seed)
        / "best_model.pt"
    )


def load_m3_predictions(
    paths: Mapping[str, Path],
    data: Mapping[str, Any],
    seeds: Sequence[int],
    device: torch.device,
    subset: str,
) -> Tuple[
    List[Dict[str, Any]],
    List[pd.DataFrame],
]:
    if subset == "validation":
        positions = data[
            "validation_positions"
        ]

        features = data[
            "X_m3_development"
        ][positions]

        occurrence = data[
            "y_occ_development"
        ][positions]

        cost = data[
            "y_cost_development"
        ][positions]

        metadata = (
            data["development_m3"]
            .iloc[positions]
            .reset_index(
                drop=True
            )
        )
    elif subset == "temporal":
        features = data[
            "X_m3_temporal"
        ]

        occurrence = data[
            "y_occ_temporal"
        ]

        cost = data[
            "y_cost_temporal"
        ]

        metadata = (
            data["temporal_m3"]
            .reset_index(
                drop=True
            )
        )
    else:
        raise ValueError(subset)

    metric_rows = []
    prediction_frames = []

    for seed in seeds:
        checkpoint_file = (
            m3_checkpoint_file(
                paths,
                int(seed),
            )
        )

        if not checkpoint_file.exists():
            raise FileNotFoundError(
                checkpoint_file
            )

        model, checkpoint = load_model(
            checkpoint_file,
            device,
        )

        predictions = predict_model(
            model,
            checkpoint,
            features,
            device,
        )

        metrics = calculate_metrics(
            occurrence,
            cost,
            predictions,
        )

        metric_rows.append(
            {
                "feature_block": "M3",
                "seed": int(seed),
                "input_dim": int(
                    checkpoint[
                        "input_dim"
                    ]
                ),
                "parameter_count": int(
                    count_parameters(
                        model
                    )
                ),
                **metrics,
            }
        )

        prediction_frames.append(
            prediction_frame(
                metadata,
                occurrence,
                cost,
                predictions,
            )
        )

    return (
        metric_rows,
        prediction_frames,
    )


# =============================================================================
# Audit stage
# =============================================================================

def run_audit(
    paths: Mapping[str, Path],
    data: Mapping[str, Any],
    profile_cfg: Mapping[str, Any],
    device: torch.device,
) -> None:
    print()
    print("=" * 100)
    print("STAGE 1 — DATA ADEQUACY AND TEMPORAL-SHIFT AUDIT")
    print("=" * 100)

    audit_dir = (
        paths["output_root"]
        / "audit"
    )

    plots_dir = (
        paths["output_root"]
        / "plots"
    )

    audit_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plots_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train = data[
        "train_positions"
    ]

    validation = data[
        "validation_positions"
    ]

    summary = pd.DataFrame(
        [
            dataset_summary(
                "development_train",
                data[
                    "development_m4"
                ].iloc[train],
            ),
            dataset_summary(
                "development_validation",
                data[
                    "development_m4"
                ].iloc[validation],
            ),
            dataset_summary(
                "temporal_2018_to_2019",
                data[
                    "temporal_m4"
                ],
            ),
        ]
    )

    summary.to_csv(
        audit_dir
        / "audit_dataset_target_summary.csv",
        index=False,
    )

    feature_names = data[
        "m4_features"
    ]

    train_frame = data[
        "development_m4"
    ].iloc[train]

    validation_frame = data[
        "development_m4"
    ].iloc[validation]

    temporal_frame = data[
        "temporal_m4"
    ]

    missing_zero_rows = []

    for cohort_name, frame in [
        (
            "development_train",
            train_frame,
        ),
        (
            "development_validation",
            validation_frame,
        ),
        (
            "temporal_2018_to_2019",
            temporal_frame,
        ),
    ]:
        for feature in feature_names:
            series = frame[feature]

            numeric = pd.to_numeric(
                series,
                errors="coerce",
            )

            missing_zero_rows.append(
                {
                    "cohort": (
                        cohort_name
                    ),
                    "feature": feature,
                    "dtype": str(
                        series.dtype
                    ),
                    "missing_rate": float(
                        series.isna().mean()
                    ),
                    "numeric_parse_rate": float(
                        numeric.notna().mean()
                    ),
                    "zero_rate_if_numeric": float(
                        (
                            numeric == 0
                        ).mean()
                    ),
                    "unique_values": int(
                        series.nunique(
                            dropna=True
                        )
                    ),
                }
            )

    pd.DataFrame(
        missing_zero_rows
    ).to_csv(
        audit_dir
        / "audit_missing_zero_unique_summary.csv",
        index=False,
    )

    numeric_rows = []
    categorical_rows = []

    for comparison_name, comparison_frame in [
        (
            "development_validation",
            validation_frame,
        ),
        (
            "temporal_2018_to_2019",
            temporal_frame,
        ),
    ]:
        for feature in data[
            "numeric_columns"
        ]:
            reference = train_frame[
                feature
            ]

            comparison = comparison_frame[
                feature
            ]

            psi = psi_numeric(
                reference,
                comparison,
            )

            smd = (
                standardized_mean_difference(
                    reference,
                    comparison,
                )
            )

            if pd.isna(psi):
                drift_flag = "not_available"
            elif psi < 0.10:
                drift_flag = "low"
            elif psi < 0.25:
                drift_flag = "moderate"
            else:
                drift_flag = "high"

            numeric_rows.append(
                {
                    "comparison": (
                        comparison_name
                    ),
                    "feature": feature,
                    "psi": psi,
                    "absolute_smd": float(
                        abs(smd)
                    ),
                    "signed_smd": smd,
                    "reference_missing_rate": float(
                        reference.isna().mean()
                    ),
                    "comparison_missing_rate": float(
                        comparison.isna().mean()
                    ),
                    "reference_mean": float(
                        pd.to_numeric(
                            reference,
                            errors="coerce",
                        ).mean()
                    ),
                    "comparison_mean": float(
                        pd.to_numeric(
                            comparison,
                            errors="coerce",
                        ).mean()
                    ),
                    "drift_flag_heuristic": (
                        drift_flag
                    ),
                }
            )

        for feature in data[
            "categorical_columns"
        ]:
            drift = categorical_drift(
                train_frame[
                    feature
                ],
                comparison_frame[
                    feature
                ],
            )

            categorical_rows.append(
                {
                    "comparison": (
                        comparison_name
                    ),
                    "feature": feature,
                    **drift,
                }
            )

    numeric_drift = pd.DataFrame(
        numeric_rows
    ).sort_values(
        [
            "comparison",
            "psi",
        ],
        ascending=[
            True,
            False,
        ],
    )

    categorical_drift_frame = (
        pd.DataFrame(
            categorical_rows
        ).sort_values(
            [
                "comparison",
                "total_variation_distance",
            ],
            ascending=[
                True,
                False,
            ],
        )
    )

    numeric_drift.to_csv(
        audit_dir
        / "audit_numeric_temporal_drift.csv",
        index=False,
    )

    categorical_drift_frame.to_csv(
        audit_dir
        / "audit_categorical_temporal_drift.csv",
        index=False,
    )

    # Tail-error contribution using the already frozen M3 Gamma-MTL ensemble.
    (
        _,
        validation_m3_frames,
    ) = load_m3_predictions(
        paths,
        data,
        profile_cfg["seeds"],
        device,
        subset="validation",
    )

    (
        _,
        temporal_m3_frames,
    ) = load_m3_predictions(
        paths,
        data,
        profile_cfg["seeds"],
        device,
        subset="temporal",
    )

    validation_ensemble = ensemble_frames(
        validation_m3_frames
    )

    temporal_ensemble = ensemble_frames(
        temporal_m3_frames
    )

    tail_table = pd.concat(
        [
            tail_error_table(
                "development_validation",
                validation_ensemble[
                    "target_cost"
                ].to_numpy(
                    dtype=float
                ),
                validation_ensemble[
                    "predicted_soft_cost"
                ].to_numpy(
                    dtype=float
                ),
            ),
            tail_error_table(
                "temporal_2018_to_2019",
                temporal_ensemble[
                    "target_cost"
                ].to_numpy(
                    dtype=float
                ),
                temporal_ensemble[
                    "predicted_soft_cost"
                ].to_numpy(
                    dtype=float
                ),
            ),
        ],
        ignore_index=True,
    )

    tail_table.to_csv(
        audit_dir
        / "audit_tail_error_contribution_m3_gamma.csv",
        index=False,
    )

    # Audit interpretation summary.
    temporal_numeric = numeric_drift.loc[
        numeric_drift[
            "comparison"
        ]
        == "temporal_2018_to_2019"
    ]

    temporal_categorical = (
        categorical_drift_frame.loc[
            categorical_drift_frame[
                "comparison"
            ]
            == "temporal_2018_to_2019"
        ]
    )

    audit_summary = {
        "high_numeric_drift_features_psi_ge_0_25": (
            temporal_numeric.loc[
                temporal_numeric["psi"]
                >= 0.25,
                "feature",
            ].tolist()
        ),
        "moderate_numeric_drift_features_psi_0_10_to_0_25": (
            temporal_numeric.loc[
                (
                    temporal_numeric["psi"]
                    >= 0.10
                )
                & (
                    temporal_numeric["psi"]
                    < 0.25
                ),
                "feature",
            ].tolist()
        ),
        "categorical_unseen_rate_gt_1pct": (
            temporal_categorical.loc[
                temporal_categorical[
                    "unseen_category_rate"
                ]
                > 0.01,
                "feature",
            ].tolist()
        ),
        "temporal_top_1pct_sse_share_pct": float(
            tail_table.loc[
                (
                    tail_table["cohort"]
                    == "temporal_2018_to_2019"
                )
                & (
                    tail_table["tail"]
                    == "top_1pct"
                ),
                "squared_error_share_pct",
            ].iloc[0]
        ),
        "note": (
            "PSI thresholds are descriptive heuristics, not formal tests. "
            "The tail table quantifies whether a small high-cost subgroup "
            "dominates RMSE."
        ),
    }

    save_json(
        audit_summary,
        audit_dir
        / "audit_interpretation_summary.json",
    )

    # Plots.
    top_numeric = (
        temporal_numeric
        .sort_values(
            "psi",
            ascending=False,
        )
        .head(15)
        .sort_values("psi")
    )

    if not top_numeric.empty:
        plt.figure(
            figsize=(9.0, 6.0)
        )

        plt.barh(
            top_numeric["feature"],
            top_numeric["psi"],
        )

        plt.axvline(
            0.10,
            linewidth=1,
        )

        plt.axvline(
            0.25,
            linewidth=1,
        )

        plt.xlabel(
            "Population Stability Index"
        )

        plt.title(
            "Largest numeric shifts: development train vs temporal cohort"
        )

        plt.tight_layout()

        plt.savefig(
            plots_dir
            / "audit_numeric_psi_top15.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

    temporal_tail = tail_table.loc[
        tail_table["cohort"]
        == "temporal_2018_to_2019"
    ]

    plt.figure(
        figsize=(7.5, 4.8)
    )

    plt.bar(
        temporal_tail["tail"],
        temporal_tail[
            "squared_error_share_pct"
        ],
    )

    plt.ylabel(
        "Share of total squared error (%)"
    )

    plt.title(
        "How much of temporal RMSE comes from the high-cost tail?"
    )

    plt.tight_layout()

    plt.savefig(
        plots_dir
        / "audit_temporal_tail_sse_share.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Audit outputs: {}".format(
            audit_dir
        )
    )


# =============================================================================
# Development M3 vs M4
# =============================================================================

def m4_run_dir(
    output_root: Path,
    seed: int,
) -> Path:
    return (
        output_root
        / "development"
        / "runs"
        / "m4"
        / "seed_{}".format(seed)
    )


def summarize_runs(
    runs: pd.DataFrame,
) -> pd.DataFrame:
    metrics = [
        "roc_auc",
        "pr_auc",
        "brier_score",
        "soft_cost_mae",
        "soft_cost_rmse",
        "soft_abs_portfolio_error_pct",
        "positive_severity_rmse",
        "positive_severity_gamma_deviance",
        "input_dim",
        "parameter_count",
    ]

    rows = []

    for feature_block, group in runs.groupby(
        "feature_block",
        sort=False,
    ):
        row: Dict[str, Any] = {
            "feature_block": (
                feature_block
            ),
            "runs": int(
                len(group)
            ),
        }

        for metric in metrics:
            values = pd.to_numeric(
                group[metric],
                errors="coerce",
            )

            row[
                metric + "_mean"
            ] = float(
                values.mean()
            )

            row[
                metric + "_std"
            ] = float(
                values.std(
                    ddof=1
                )
                if len(values) > 1
                else 0.0
            )

        rows.append(row)

    return pd.DataFrame(rows)


def run_development(
    paths: Mapping[str, Path],
    data: Mapping[str, Any],
    profile_cfg: Mapping[str, Any],
    device: torch.device,
    force: bool,
) -> Dict[str, Any]:
    print()
    print("=" * 100)
    print("STAGE 2 — CONTROLLED M3 VERSUS M4 DEVELOPMENT ABLATION")
    print("=" * 100)

    development_dir = (
        paths["output_root"]
        / "development"
    )

    tables_dir = (
        paths["output_root"]
        / "paper_tables"
    )

    plots_dir = (
        paths["output_root"]
        / "plots"
    )

    for directory in [
        development_dir,
        tables_dir,
        plots_dir,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    joblib.dump(
        data["m4_preprocessor"],
        development_dir
        / "m4_preprocessor.joblib",
    )

    save_json(
        data["m4_context"],
        development_dir
        / "m4_feature_context.json",
    )

    save_json(
        {
            "m3_features": (
                data["m3_features"]
            ),
            "m4_added_features": (
                M4_FEATURE_NAMES
            ),
            "m4_all_features": (
                data["m4_features"]
            ),
            "numeric_columns": (
                data["numeric_columns"]
            ),
            "categorical_columns": (
                data[
                    "categorical_columns"
                ]
            ),
            "m3_encoded_dim": int(
                data[
                    "X_m3_development"
                ].shape[1]
            ),
            "m4_encoded_dim": int(
                data[
                    "X_m4_development"
                ].shape[1]
            ),
        },
        development_dir
        / "feature_block_definition.json",
    )

    train = data[
        "train_positions"
    ]

    validation = data[
        "validation_positions"
    ]

    # Frozen M3 baseline.
    (
        m3_metric_rows,
        m3_prediction_frames,
    ) = load_m3_predictions(
        paths,
        data,
        profile_cfg["seeds"],
        device,
        subset="validation",
    )

    # New M4 runs.
    m4_metric_rows = []
    m4_prediction_frames = []

    for seed in profile_cfg[
        "seeds"
    ]:
        seed = int(seed)

        run_dir = m4_run_dir(
            paths["output_root"],
            seed,
        )

        checkpoint_file = (
            run_dir
            / "best_model.pt"
        )

        metrics_file = (
            run_dir
            / "validation_metrics.json"
        )

        predictions_file = (
            run_dir
            / "validation_predictions.csv"
        )

        if (
            checkpoint_file.exists()
            and metrics_file.exists()
            and predictions_file.exists()
            and not force
        ):
            print(
                "Resume M4 seed {}".format(
                    seed
                ),
                flush=True,
            )

            m4_metric_rows.append(
                load_json(
                    metrics_file
                )
            )

            m4_prediction_frames.append(
                pd.read_csv(
                    predictions_file
                )
            )

            continue

        model, checkpoint = train_m4_seed(
            seed=seed,
            X_train=data[
                "X_m4_development"
            ][train],
            y_occ_train=data[
                "y_occ_development"
            ][train],
            y_cost_train=data[
                "y_cost_development"
            ][train],
            X_validation=data[
                "X_m4_development"
            ][validation],
            y_occ_validation=data[
                "y_occ_development"
            ][validation],
            y_cost_validation=data[
                "y_cost_development"
            ][validation],
            gamma_scale=float(
                data["gamma_scale"]
            ),
            profile_cfg=profile_cfg,
            device=device,
            run_dir=run_dir,
        )

        predictions = predict_model(
            model,
            checkpoint,
            data[
                "X_m4_development"
            ][validation],
            device,
        )

        metrics = calculate_metrics(
            data[
                "y_occ_development"
            ][validation],
            data[
                "y_cost_development"
            ][validation],
            predictions,
        )

        payload = {
            "feature_block": "M4",
            "seed": seed,
            "input_dim": int(
                checkpoint[
                    "input_dim"
                ]
            ),
            "parameter_count": int(
                checkpoint[
                    "parameter_count"
                ]
            ),
            "best_epoch": int(
                checkpoint[
                    "best_epoch"
                ]
            ),
            "best_validation_loss": float(
                checkpoint[
                    "best_validation_loss"
                ]
            ),
            **metrics,
        }

        save_json(
            payload,
            metrics_file,
        )

        frame = prediction_frame(
            data[
                "development_m4"
            ].iloc[
                validation
            ].reset_index(
                drop=True
            ),
            data[
                "y_occ_development"
            ][validation],
            data[
                "y_cost_development"
            ][validation],
            predictions,
        )

        frame.to_csv(
            predictions_file,
            index=False,
        )

        m4_metric_rows.append(
            payload
        )

        m4_prediction_frames.append(
            frame
        )

    runs = pd.DataFrame(
        m3_metric_rows
        + m4_metric_rows
    )

    runs.to_csv(
        development_dir
        / "development_m3_vs_m4_all_runs.csv",
        index=False,
    )

    summary = summarize_runs(
        runs
    )

    summary.to_csv(
        development_dir
        / "development_m3_vs_m4_summary.csv",
        index=False,
    )

    summary.to_csv(
        tables_dir
        / "table_01_m3_vs_m4_development.csv",
        index=False,
    )

    m3_seed = runs.loc[
        runs["feature_block"]
        == "M3"
    ].set_index("seed")

    m4_seed = runs.loc[
        runs["feature_block"]
        == "M4"
    ].set_index("seed")

    common_seeds = sorted(
        set(m3_seed.index)
        & set(m4_seed.index)
    )

    paired_rows = []

    metrics = [
        "roc_auc",
        "brier_score",
        "soft_cost_mae",
        "soft_cost_rmse",
        "positive_severity_rmse",
        "soft_abs_portfolio_error_pct",
    ]

    for seed in common_seeds:
        row: Dict[str, Any] = {
            "seed": int(seed),
        }

        for metric in metrics:
            difference = float(
                m4_seed.loc[
                    seed,
                    metric,
                ]
                - m3_seed.loc[
                    seed,
                    metric,
                ]
            )

            row[
                "m4_minus_m3_"
                + metric
            ] = difference

            if metric == "roc_auc":
                row[
                    "m4_better_"
                    + metric
                ] = bool(
                    difference > 0
                )
            else:
                row[
                    "m4_better_"
                    + metric
                ] = bool(
                    difference < 0
                )

        paired_rows.append(row)

    paired = pd.DataFrame(
        paired_rows
    )

    paired.to_csv(
        development_dir
        / "development_m4_minus_m3_paired.csv",
        index=False,
    )

    paired.to_csv(
        tables_dir
        / "table_02_m4_minus_m3_paired_development.csv",
        index=False,
    )

    m3_summary = summary.loc[
        summary["feature_block"]
        == "M3"
    ].iloc[0]

    m4_summary = summary.loc[
        summary["feature_block"]
        == "M4"
    ].iloc[0]

    rmse_difference = float(
        m4_summary[
            "soft_cost_rmse_mean"
        ]
        - m3_summary[
            "soft_cost_rmse_mean"
        ]
    )

    roc_difference = float(
        m4_summary[
            "roc_auc_mean"
        ]
        - m3_summary[
            "roc_auc_mean"
        ]
    )

    rmse_wins = int(
        paired[
            "m4_better_soft_cost_rmse"
        ].sum()
    )

    required_wins = max(
        1,
        math.ceil(
            0.75
            * len(common_seeds)
        ),
    )

    promotion = {
        "selection_data": (
            "2017->2018 development validation only"
        ),
        "reference": "M3",
        "candidate": "M4",
        "primary_metric": (
            PRIMARY_METRIC
        ),
        "m4_minus_m3_soft_cost_rmse": (
            rmse_difference
        ),
        "mean_rmse_improved": bool(
            rmse_difference < 0
        ),
        "paired_rmse_wins": (
            rmse_wins
        ),
        "required_paired_wins": (
            required_wins
        ),
        "consistency_met": bool(
            rmse_wins
            >= required_wins
        ),
        "m4_minus_m3_roc_auc": (
            roc_difference
        ),
        "roc_guardrail_tolerance": (
            ROC_GUARDRAIL_TOLERANCE
        ),
        "roc_guardrail_met": bool(
            roc_difference
            >= -ROC_GUARDRAIL_TOLERANCE
        ),
    }

    promotion[
        "promotion_status"
    ] = (
        "promote_m4"
        if (
            promotion[
                "mean_rmse_improved"
            ]
            and promotion[
                "consistency_met"
            ]
            and promotion[
                "roc_guardrail_met"
            ]
        )
        else "keep_m3"
    )

    save_json(
        promotion,
        development_dir
        / "m4_promotion_decision.json",
    )

    plot_table = summary.sort_values(
        "soft_cost_rmse_mean",
        ascending=True,
    )

    plt.figure(
        figsize=(7.2, 4.4)
    )

    plt.barh(
        plot_table[
            "feature_block"
        ],
        plot_table[
            "soft_cost_rmse_mean"
        ],
        xerr=plot_table[
            "soft_cost_rmse_std"
        ],
    )

    plt.xlabel(
        "Development soft-cost RMSE"
    )

    plt.title(
        "Final feature ablation: M3 versus M4"
    )

    plt.tight_layout()

    plt.savefig(
        plots_dir
        / "development_m3_vs_m4_rmse.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print()
    print(
        summary[
            [
                "feature_block",
                "input_dim_mean",
                "parameter_count_mean",
                "roc_auc_mean",
                "soft_cost_mae_mean",
                "soft_cost_rmse_mean",
                "positive_severity_rmse_mean",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Promotion decision: {}".format(
            promotion[
                "promotion_status"
            ]
        )
    )

    return {
        "runs": runs,
        "summary": summary,
        "paired": paired,
        "promotion": promotion,
        "m3_prediction_frames": (
            m3_prediction_frames
        ),
        "m4_prediction_frames": (
            m4_prediction_frames
        ),
    }


# =============================================================================
# Policy bootstrap
# =============================================================================

def paired_policy_bootstrap(
    m3_ensemble: pd.DataFrame,
    m4_ensemble: pd.DataFrame,
    replicates: int,
    random_seed: int = 20260721,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    key_columns = [
        "row_id",
        "ID_policy",
        "target_occurrence",
        "target_cost",
    ]

    for column in key_columns:
        if not np.array_equal(
            m3_ensemble[column].to_numpy(),
            m4_ensemble[column].to_numpy(),
        ):
            raise RuntimeError(
                "M3 and M4 ensembles are not aligned."
            )

    actual_occurrence = (
        m3_ensemble[
            "target_occurrence"
        ].to_numpy(
            dtype=float
        )
    )

    actual_cost = (
        m3_ensemble[
            "target_cost"
        ].to_numpy(
            dtype=float
        )
    )

    positive = (
        actual_occurrence == 1
    )

    table = m3_ensemble[
        key_columns
    ].copy()

    table["rows_n"] = 1
    table["positive_n"] = (
        positive.astype(int)
    )

    for model_name, frame in [
        (
            "m3",
            m3_ensemble,
        ),
        (
            "m4",
            m4_ensemble,
        ),
    ]:
        probability = frame[
            "predicted_probability"
        ].to_numpy(
            dtype=float
        )

        severity = frame[
            "predicted_positive_cost"
        ].to_numpy(
            dtype=float
        )

        soft_cost = frame[
            "predicted_soft_cost"
        ].to_numpy(
            dtype=float
        )

        table[
            model_name
            + "_soft_sse"
        ] = (
            actual_cost
            - soft_cost
        ) ** 2

        table[
            model_name
            + "_soft_sae"
        ] = np.abs(
            actual_cost
            - soft_cost
        )

        table[
            model_name
            + "_brier_sse"
        ] = (
            actual_occurrence
            - probability
        ) ** 2

        table[
            model_name
            + "_severity_sse"
        ] = np.where(
            positive,
            (
                actual_cost
                - severity
            ) ** 2,
            0.0,
        )

    statistic_columns = [
        column
        for column in table.columns
        if column not in key_columns
    ]

    policy = (
        table.groupby(
            "ID_policy",
            as_index=False,
        )[
            statistic_columns
        ]
        .sum()
    )

    values = policy[
        statistic_columns
    ].to_numpy(
        dtype=float
    )

    column_index = {
        name: index
        for index, name in enumerate(
            statistic_columns
        )
    }

    rng = np.random.default_rng(
        random_seed
    )

    rows = []

    for replicate in range(
        1,
        replicates + 1,
    ):
        sampled = rng.integers(
            0,
            len(policy),
            size=len(policy),
        )

        counts = np.bincount(
            sampled,
            minlength=len(policy),
        ).astype(float)

        totals = counts @ values

        rows_n = totals[
            column_index[
                "rows_n"
            ]
        ]

        positive_n = totals[
            column_index[
                "positive_n"
            ]
        ]

        metrics = {}

        for model_name in [
            "m3",
            "m4",
        ]:
            metrics[
                model_name
                + "_rmse"
            ] = math.sqrt(
                totals[
                    column_index[
                        model_name
                        + "_soft_sse"
                    ]
                ]
                / rows_n
            )

            metrics[
                model_name
                + "_mae"
            ] = (
                totals[
                    column_index[
                        model_name
                        + "_soft_sae"
                    ]
                ]
                / rows_n
            )

            metrics[
                model_name
                + "_brier"
            ] = (
                totals[
                    column_index[
                        model_name
                        + "_brier_sse"
                    ]
                ]
                / rows_n
            )

            metrics[
                model_name
                + "_severity_rmse"
            ] = math.sqrt(
                totals[
                    column_index[
                        model_name
                        + "_severity_sse"
                    ]
                ]
                / positive_n
            )

        rows.append(
            {
                "replicate": replicate,
                "m4_minus_m3_soft_rmse": (
                    metrics[
                        "m4_rmse"
                    ]
                    - metrics[
                        "m3_rmse"
                    ]
                ),
                "m4_minus_m3_soft_mae": (
                    metrics[
                        "m4_mae"
                    ]
                    - metrics[
                        "m3_mae"
                    ]
                ),
                "m4_minus_m3_brier": (
                    metrics[
                        "m4_brier"
                    ]
                    - metrics[
                        "m3_brier"
                    ]
                ),
                "m4_minus_m3_severity_rmse": (
                    metrics[
                        "m4_severity_rmse"
                    ]
                    - metrics[
                        "m3_severity_rmse"
                    ]
                ),
            }
        )

    replicates_frame = pd.DataFrame(
        rows
    )

    summary_rows = []

    for metric in [
        column
        for column in replicates_frame.columns
        if column.startswith(
            "m4_minus_m3_"
        )
    ]:
        values_metric = (
            replicates_frame[
                metric
            ].to_numpy(
                dtype=float
            )
        )

        summary_rows.append(
            {
                "metric": metric,
                "mean_difference": float(
                    values_metric.mean()
                ),
                "ci_2_5": float(
                    np.quantile(
                        values_metric,
                        0.025,
                    )
                ),
                "ci_97_5": float(
                    np.quantile(
                        values_metric,
                        0.975,
                    )
                ),
                "probability_m4_better": float(
                    np.mean(
                        values_metric < 0
                    )
                ),
                "bootstrap_replicates": int(
                    replicates
                ),
                "bootstrap_unit": (
                    "ID_policy"
                ),
            }
        )

    return (
        replicates_frame,
        pd.DataFrame(
            summary_rows
        ),
    )


# =============================================================================
# Temporal stage
# =============================================================================

def write_freeze_record(
    paths: Mapping[str, Path],
    data: Mapping[str, Any],
    profile_cfg: Mapping[str, Any],
    promotion: Mapping[str, Any],
) -> Path:
    checkpoints = []

    for seed in profile_cfg[
        "seeds"
    ]:
        checkpoint = (
            m4_run_dir(
                paths[
                    "output_root"
                ],
                int(seed),
            )
            / "best_model.pt"
        )

        if not checkpoint.exists():
            raise FileNotFoundError(
                checkpoint
            )

        checkpoints.append(
            {
                "seed": int(seed),
                "path": checkpoint,
                "sha256": sha256_file(
                    checkpoint
                ),
            }
        )

    record = {
        "created_before_temporal_metrics": True,
        "candidate": "M4 Shared Gamma-MTL",
        "reference": "M3 Shared Gamma-MTL",
        "development_promotion_decision": (
            promotion
        ),
        "architecture": (
            "input->128->64 shared, 64->32->1 per head"
        ),
        "loss": (
            "BCE + 0.10 Gamma mean NLL"
        ),
        "learning_rate": (
            LEARNING_RATE
        ),
        "weight_decay": (
            WEIGHT_DECAY
        ),
        "epochs": int(
            profile_cfg["epochs"]
        ),
        "seeds": list(
            profile_cfg["seeds"]
        ),
        "m4_features": (
            data["m4_features"]
        ),
        "m4_added_features": (
            M4_FEATURE_NAMES
        ),
        "m4_context": (
            data["m4_context"]
        ),
        "m4_checkpoints": (
            checkpoints
        ),
        "development_sha256": (
            sha256_file(
                paths[
                    "development_file"
                ]
            )
        ),
        "temporal_sha256": (
            sha256_file(
                paths[
                    "temporal_file"
                ]
            )
        ),
        "reporting_boundary": (
            "2018->2019 was already inspected in earlier experiments. "
            "This is a frozen post-hoc out-of-time benchmark."
        ),
    }

    path = (
        paths["output_root"]
        / "freeze_record_before_temporal.json"
    )

    save_json(
        record,
        path,
    )

    return path


def run_temporal(
    paths: Mapping[str, Path],
    data: Mapping[str, Any],
    profile_cfg: Mapping[str, Any],
    device: torch.device,
    force_temporal: bool,
) -> Optional[
    Dict[str, Any]
]:
    print()
    print("=" * 100)
    print("STAGE 3 — FROZEN M3 VERSUS M4 TEMPORAL BENCHMARK")
    print("=" * 100)

    decision_file = (
        paths["output_root"]
        / "development"
        / "m4_promotion_decision.json"
    )

    if not decision_file.exists():
        raise FileNotFoundError(
            "Run development stage first."
        )

    promotion = load_json(
        decision_file
    )

    if (
        promotion[
            "promotion_status"
        ]
        != "promote_m4"
        and not force_temporal
    ):
        print(
            "M4 did not meet the predeclared development promotion rule."
        )
        print(
            "Temporal M4 evaluation was not run. The final model remains M3."
        )

        save_json(
            {
                "status": (
                    "temporal_not_run"
                ),
                "reason": (
                    "M4 failed development promotion rule"
                ),
                "promotion": promotion,
            },
            paths[
                "output_root"
            ]
            / "temporal"
            / "temporal_stage_status.json",
        )

        return None

    temporal_dir = (
        paths["output_root"]
        / "temporal"
    )

    tables_dir = (
        paths["output_root"]
        / "paper_tables"
    )

    plots_dir = (
        paths["output_root"]
        / "plots"
    )

    for directory in [
        temporal_dir,
        tables_dir,
        plots_dir,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    freeze_path = write_freeze_record(
        paths,
        data,
        profile_cfg,
        promotion,
    )

    print(
        "Freeze record: {}".format(
            freeze_path
        )
    )

    (
        m3_metric_rows,
        m3_frames,
    ) = load_m3_predictions(
        paths,
        data,
        profile_cfg["seeds"],
        device,
        subset="temporal",
    )

    m4_metric_rows = []
    m4_frames = []

    for seed in profile_cfg[
        "seeds"
    ]:
        seed = int(seed)

        checkpoint_file = (
            m4_run_dir(
                paths["output_root"],
                seed,
            )
            / "best_model.pt"
        )

        model, checkpoint = load_model(
            checkpoint_file,
            device,
        )

        predictions = predict_model(
            model,
            checkpoint,
            data[
                "X_m4_temporal"
            ],
            device,
        )

        metrics = calculate_metrics(
            data[
                "y_occ_temporal"
            ],
            data[
                "y_cost_temporal"
            ],
            predictions,
        )

        m4_metric_rows.append(
            {
                "feature_block": "M4",
                "seed": seed,
                "input_dim": int(
                    checkpoint[
                        "input_dim"
                    ]
                ),
                "parameter_count": int(
                    count_parameters(
                        model
                    )
                ),
                **metrics,
            }
        )

        frame = prediction_frame(
            data[
                "temporal_m4"
            ],
            data[
                "y_occ_temporal"
            ],
            data[
                "y_cost_temporal"
            ],
            predictions,
        )

        m4_frames.append(frame)

        seed_dir = (
            temporal_dir
            / "seed_{}".format(seed)
        )

        seed_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        frame.to_csv(
            seed_dir
            / "m4_predictions.csv",
            index=False,
        )

    runs = pd.DataFrame(
        m3_metric_rows
        + m4_metric_rows
    )

    runs.to_csv(
        temporal_dir
        / "temporal_m3_vs_m4_all_runs.csv",
        index=False,
    )

    summary = summarize_runs(
        runs
    )

    summary.to_csv(
        temporal_dir
        / "temporal_m3_vs_m4_summary.csv",
        index=False,
    )

    summary.to_csv(
        tables_dir
        / "table_03_m3_vs_m4_temporal.csv",
        index=False,
    )

    m3_seed = runs.loc[
        runs["feature_block"]
        == "M3"
    ].set_index("seed")

    m4_seed = runs.loc[
        runs["feature_block"]
        == "M4"
    ].set_index("seed")

    paired_rows = []

    for seed in sorted(
        set(m3_seed.index)
        & set(m4_seed.index)
    ):
        row = {
            "seed": int(seed),
        }

        for metric in [
            "roc_auc",
            "brier_score",
            "soft_cost_mae",
            "soft_cost_rmse",
            "positive_severity_rmse",
            "soft_abs_portfolio_error_pct",
        ]:
            row[
                "m4_minus_m3_"
                + metric
            ] = float(
                m4_seed.loc[
                    seed,
                    metric,
                ]
                - m3_seed.loc[
                    seed,
                    metric,
                ]
            )

        paired_rows.append(row)

    paired = pd.DataFrame(
        paired_rows
    )

    paired.to_csv(
        temporal_dir
        / "temporal_m4_minus_m3_paired.csv",
        index=False,
    )

    paired.to_csv(
        tables_dir
        / "table_04_m4_minus_m3_paired_temporal.csv",
        index=False,
    )

    m3_ensemble = ensemble_frames(
        m3_frames
    )

    m4_ensemble = ensemble_frames(
        m4_frames
    )

    m3_ensemble.to_csv(
        temporal_dir
        / "m3_ensemble_predictions.csv",
        index=False,
    )

    m4_ensemble.to_csv(
        temporal_dir
        / "m4_ensemble_predictions.csv",
        index=False,
    )

    (
        bootstrap_replicates,
        bootstrap_summary,
    ) = paired_policy_bootstrap(
        m3_ensemble,
        m4_ensemble,
        replicates=int(
            profile_cfg[
                "bootstrap_replicates"
            ]
        ),
    )

    bootstrap_replicates.to_csv(
        temporal_dir
        / "policy_bootstrap_replicates.csv",
        index=False,
    )

    bootstrap_summary.to_csv(
        temporal_dir
        / "policy_bootstrap_summary.csv",
        index=False,
    )

    bootstrap_summary.to_csv(
        tables_dir
        / "table_05_m4_vs_m3_policy_bootstrap.csv",
        index=False,
    )

    plot_table = summary.sort_values(
        "soft_cost_rmse_mean",
        ascending=True,
    )

    plt.figure(
        figsize=(7.2, 4.4)
    )

    plt.barh(
        plot_table[
            "feature_block"
        ],
        plot_table[
            "soft_cost_rmse_mean"
        ],
        xerr=plot_table[
            "soft_cost_rmse_std"
        ],
    )

    plt.xlabel(
        "Temporal soft-cost RMSE"
    )

    plt.title(
        "Frozen post-hoc temporal feature comparison"
    )

    plt.tight_layout()

    plt.savefig(
        plots_dir
        / "temporal_m3_vs_m4_rmse.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print()
    print(
        summary[
            [
                "feature_block",
                "roc_auc_mean",
                "brier_score_mean",
                "soft_cost_mae_mean",
                "soft_cost_rmse_mean",
                "positive_severity_rmse_mean",
            ]
        ].to_string(
            index=False
        )
    )

    return {
        "runs": runs,
        "summary": summary,
        "paired": paired,
        "bootstrap_summary": (
            bootstrap_summary
        ),
    }


# =============================================================================
# Protocol and README
# =============================================================================

def save_protocol(
    paths: Mapping[str, Path],
    data: Mapping[str, Any],
    args: argparse.Namespace,
    profile_cfg: Mapping[str, Any],
) -> None:
    payload = {
        "study": (
            "Final data audit and M4 feature ablation "
            "for Shared Gamma-MTL"
        ),
        "profile": args.profile,
        "stage": args.stage,
        "seeds": list(
            profile_cfg["seeds"]
        ),
        "epochs": int(
            profile_cfg["epochs"]
        ),
        "architecture": (
            "input->128->64 shared, 64->32->1 per head"
        ),
        "loss": (
            "BCE + 0.10 Gamma mean NLL"
        ),
        "learning_rate": (
            LEARNING_RATE
        ),
        "weight_decay": (
            WEIGHT_DECAY
        ),
        "development_split": {
            "method": (
                "GroupShuffleSplit"
            ),
            "group": (
                "ID_policy"
            ),
            "validation_size": (
                VALIDATION_SIZE
            ),
            "random_state": (
                SPLIT_RANDOM_STATE
            ),
        },
        "m3_feature_count_raw": int(
            len(
                data["m3_features"]
            )
        ),
        "m4_feature_count_raw": int(
            len(
                data["m4_features"]
            )
        ),
        "m3_encoded_dim": int(
            data[
                "X_m3_development"
            ].shape[1]
        ),
        "m4_encoded_dim": int(
            data[
                "X_m4_development"
            ].shape[1]
        ),
        "m4_added_features": (
            M4_FEATURE_NAMES
        ),
        "m4_feature_context": (
            data["m4_context"]
        ),
        "gamma_scale": float(
            data["gamma_scale"]
        ),
        "temporal_reporting_boundary": (
            "2018->2019 was already inspected. Any M4 temporal result is "
            "a frozen post-hoc out-of-time benchmark."
        ),
        "software": {
            "python": (
                "{}.{}.{}".format(
                    os.sys.version_info.major,
                    os.sys.version_info.minor,
                    os.sys.version_info.micro,
                )
            ),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": (
                sklearn.__version__
            ),
            "pytorch": torch.__version__,
        },
        "source_files": {
            "development": (
                paths[
                    "development_file"
                ]
            ),
            "temporal": (
                paths[
                    "temporal_file"
                ]
            ),
            "m3_preprocessor": (
                paths[
                    "m3_preprocessor"
                ]
            ),
            "m3_checkpoint_root": (
                paths[
                    "m3_checkpoint_root"
                ]
            ),
        },
    }

    save_json(
        payload,
        paths[
            "output_root"
        ]
        / "protocol_and_integrity.json",
    )


def build_readme(
    paths: Mapping[str, Path],
) -> None:
    text = """GAMMA-MTL DATA AUDIT + M4 FINAL EXPERIMENT
================================================

SCIENTIFIC QUESTION
-------------------
Does a small actuarially motivated M4 feature block improve the already
selected Shared Gamma-MTL, or has the available tabular information reached
a practical prediction ceiling?

OUTPUT ORDER
------------
1. audit/audit_interpretation_summary.json
2. audit/audit_numeric_temporal_drift.csv
3. audit/audit_categorical_temporal_drift.csv
4. audit/audit_tail_error_contribution_m3_gamma.csv
5. paper_tables/table_01_m3_vs_m4_development.csv
6. paper_tables/table_02_m4_minus_m3_paired_development.csv
7. development/m4_promotion_decision.json

TEMPORAL OUTPUTS ARE CREATED ONLY IF M4 PASSES DEVELOPMENT
----------------------------------------------------------
8. paper_tables/table_03_m3_vs_m4_temporal.csv
9. paper_tables/table_04_m4_minus_m3_paired_temporal.csv
10. paper_tables/table_05_m4_vs_m3_policy_bootstrap.csv

DECISION RULE
-------------
Keep M4 only when:
- mean development soft-cost RMSE is lower than M3,
- M4 wins RMSE in at least 75% of seeds,
- ROC-AUC does not decrease by more than 0.002.

If the rule is not met, the final proposed model remains the existing M3
Shared Gamma-MTL. Do not continue modifying the 2018->2019 result.

REPORTING BOUNDARY
------------------
The 2018->2019 period was already used in previous experiments. Any M4
temporal comparison is a frozen post-hoc out-of-time benchmark, not a newly
untouched confirmatory test.
"""

    (
        paths["output_root"]
        / "README_RESULTS.txt"
    ).write_text(
        text,
        encoding="utf-8",
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_arguments()

    profile_cfg = PROFILES[
        args.profile
    ]

    paths = resolve_paths(
        args.project_dir,
        args.profile,
    )

    paths["output_root"].mkdir(
        parents=True,
        exist_ok=True,
    )

    device = configure_cpu()

    set_seed(
        int(
            profile_cfg[
                "seeds"
            ][0]
        )
    )

    print("=" * 100)
    print("SHARED GAMMA-MTL — DATA AUDIT AND FINAL M4 FEATURE ABLATION")
    print("=" * 100)
    print(
        "Project: {}".format(
            paths["project_dir"]
        )
    )
    print(
        "Profile: {}".format(
            args.profile
        )
    )
    print(
        "Stage:   {}".format(
            args.stage
        )
    )
    print(
        "Seeds:   {}".format(
            profile_cfg[
                "seeds"
            ]
        )
    )
    print(
        "Device:  {}".format(
            device
        )
    )

    data = prepare_data(paths)

    save_protocol(
        paths,
        data,
        args,
        profile_cfg,
    )

    stages = (
        [
            "audit",
            "development",
            "temporal",
        ]
        if args.stage == "all"
        else [args.stage]
    )

    if "audit" in stages:
        run_audit(
            paths,
            data,
            profile_cfg,
            device,
        )

    if "development" in stages:
        run_development(
            paths,
            data,
            profile_cfg,
            device,
            force=args.force,
        )

    if "temporal" in stages:
        run_temporal(
            paths,
            data,
            profile_cfg,
            device,
            force_temporal=(
                args.force_temporal
            ),
        )

    build_readme(paths)

    print()
    print("=" * 100)
    print("COMPLETE")
    print("=" * 100)
    print(
        "Outputs: {}".format(
            paths["output_root"]
        )
    )
    print(
        "Send the audit folder, paper_tables folder, "
        "m4_promotion_decision.json and terminal log."
    )


if __name__ == "__main__":
    main()
