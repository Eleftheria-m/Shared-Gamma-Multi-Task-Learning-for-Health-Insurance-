#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final fair M4 comparison and explainability pipeline.

Dependencies
------------
Place this file together with:
    gamma_mtl_audit_m4_final_FIXED.py
inside:
    /home/eleut/health_insurance_project

Scientific protocol
-------------------
All learned models use the same:
- development policy split,
- frozen M4 feature block and preprocessor,
- encoded input matrix,
- 2018->2019 out-of-time cohort,
- expected-cost output p(x) * mu(x).

Final models
------------
1. Constant two-part baseline
2. Previous-year persistence
3. Logistic + Gamma GLM on M4
4. HistGradientBoosting two-part on M4
5. M4 Shared Gamma-MTL

The M3-vs-M4 experiment remains the only feature ablation. It is copied into
this final output folder instead of repeating M3/M4 across all baselines.

Explainability
--------------
- Common grouped Monte-Carlo SHAP approximation for GLM, HGB and Gamma-MTL.
- Separate occurrence, positive-severity and soft-cost explanations.
- Seed stability with Spearman rho and top-10 Jaccard.
- Top-5 deletion fidelity versus random deletion.
- Integrated Gradients for Gamma-MTL and IG-SHAP agreement.

Reporting boundary
------------------
The 2018->2019 cohort has already been inspected in prior experiments.
Describe this as a frozen post-hoc out-of-time comparison, not as a newly
untouched confirmatory test.
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import gc
import json
import math
import random
import shutil
import tempfile
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import GammaRegressor, LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    mean_gamma_deviance,
    mean_squared_error,
    roc_auc_score,
)

try:
    from gamma_mtl_audit_m4_final_FIXED import (
        SharedGammaMTL,
        apply_m4_features,
        sanitize_sklearn_frame,
        grouped_split,
        load_model,
        predict_model,
        prediction_frame,
        ensemble_frames,
        count_parameters,
    )
except ImportError as exc:
    raise ImportError(
        "Place gamma_mtl_audit_m4_final_FIXED.py in the same project folder."
    ) from exc


PROJECT_DEFAULT = Path(
    os.environ.get("HEALTH_PROJECT_DIR", "/home/eleut/health_insurance_project")
)

PROFILES = {
    "quick": {
        "seeds": [42],
        "bootstrap": 50,
        "shap_rows": 64,
        "background_rows": 128,
        "shap_permutations": 4,
        "fidelity_rows": 256,
        "fidelity_repeats": 2,
        "ig_rows": 64,
        "ig_steps": 8,
        "ig_batch": 32,
        "runtime_warmup": 2,
        "runtime_repeats": 5,
    },
    "paper": {
        "seeds": [42, 123, 2026, 31415],
        "bootstrap": 500,
        "shap_rows": 256,
        "background_rows": 512,
        "shap_permutations": 20,
        "fidelity_rows": 2000,
        "fidelity_repeats": 10,
        "ig_rows": 256,
        "ig_steps": 32,
        "ig_batch": 64,
        "runtime_warmup": 3,
        "runtime_repeats": 20,
    },
}

GLM_C = 0.10
GAMMA_ALPHA = 0.0001
HGB_LR = 0.05
HGB_LEAVES = 15
HGB_ITER = 250
EPS = 1e-8
MIN_LOG_MEAN = -8.0
MAX_LOG_MEAN = 8.0
TASKS = ["occurrence", "severity", "soft_cost"]
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=PROJECT_DEFAULT)
    parser.add_argument("--profile", choices=PROFILES, default="paper")
    parser.add_argument(
        "--stage",
        choices=["all", "train", "test", "complexity", "explainability"],
        default="all",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def configure_cpu() -> torch.device:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return torch.device("cpu")


def save_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            default=lambda x: (
                str(x)
                if isinstance(x, Path)
                else int(x)
                if isinstance(x, np.integer)
                else float(x)
                if isinstance(x, np.floating)
                else x.tolist()
                if isinstance(x, np.ndarray)
                else None
            ),
        ),
        encoding="utf-8",
    )


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(actual, predicted)))


def portfolio_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    total = float(actual.sum())
    return float(100.0 * (predicted.sum() - total) / total) if total else np.nan


def resolve_paths(project: Path, profile: str) -> Dict[str, Path]:
    project = project.expanduser().resolve()
    m4 = project / "outputs" / "gamma_mtl_audit_m4_final" / "paper"
    paths = {
        "project": project,
        "development": project / "outputs" / "week1_step2" / "development_2017_2018_full_exposure.parquet",
        "temporal": project / "outputs" / "week1_step2" / "temporal_test_2018_2019_full_exposure.parquet",
        "m4_preprocessor": m4 / "development" / "m4_preprocessor.joblib",
        "m4_context": m4 / "development" / "m4_feature_context.json",
        "m4_definition": m4 / "development" / "feature_block_definition.json",
        "m4_checkpoints": m4 / "development" / "runs" / "m4",
        "ablation_dev": m4 / "paper_tables" / "table_01_m3_vs_m4_development.csv",
        "ablation_test": m4 / "paper_tables" / "table_03_m3_vs_m4_temporal.csv",
        "output": project / "outputs" / "final_m4_fair_comparison" / profile,
    }
    missing = [p for k, p in paths.items() if k not in {"output", "ablation_dev", "ablation_test"} and not p.exists()]
    if missing:
        raise FileNotFoundError("Missing files:\n" + "\n".join(map(str, missing)))
    return paths


def prepare_data(paths: Mapping[str, Path]) -> Dict[str, Any]:
    dev_raw = pd.read_parquet(paths["development"]).reset_index(drop=True)
    test_raw = pd.read_parquet(paths["temporal"]).reset_index(drop=True)
    context = load_json(paths["m4_context"])
    definition = load_json(paths["m4_definition"])
    preprocessor = joblib.load(paths["m4_preprocessor"])

    features = list(map(str, definition["m4_all_features"]))
    numeric = list(map(str, definition["numeric_columns"]))
    categorical = list(map(str, definition["categorical_columns"]))

    dev = apply_m4_features(dev_raw, context)
    test = apply_m4_features(test_raw, context)
    train_idx, val_idx = grouped_split(dev)

    dev_clean = sanitize_sklearn_frame(dev[features], numeric, categorical)
    test_clean = sanitize_sklearn_frame(test[features], numeric, categorical)
    X_dev = np.asarray(preprocessor.transform(dev_clean), dtype=np.float32)
    X_test = np.asarray(preprocessor.transform(test_clean), dtype=np.float32)

    if not np.isfinite(X_dev).all() or not np.isfinite(X_test).all():
        raise ValueError("Encoded M4 matrix contains non-finite values.")

    return {
        "dev": dev,
        "test": test,
        "preprocessor": preprocessor,
        "features": features,
        "X_dev": X_dev,
        "X_test": X_test,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "y_occ_dev": pd.to_numeric(dev["target_occurrence"], errors="raise").to_numpy(np.int64),
        "y_cost_dev": pd.to_numeric(dev["target_cost"], errors="raise").to_numpy(np.float64),
        "y_occ_test": pd.to_numeric(test["target_occurrence"], errors="raise").to_numpy(np.int64),
        "y_cost_test": pd.to_numeric(test["target_cost"], errors="raise").to_numpy(np.float64),
    }


def fit_glm(X: np.ndarray, y_occ: np.ndarray, y_cost: np.ndarray) -> Dict[str, Any]:
    occ = LogisticRegression(C=GLM_C, solver="liblinear", max_iter=2000, random_state=42)
    occ.fit(X, y_occ)
    positive = y_occ == 1
    sev = GammaRegressor(alpha=GAMMA_ALPHA, max_iter=2000, tol=1e-7)
    sev.fit(X[positive], y_cost[positive])
    return {"occurrence_model": occ, "severity_model": sev}


def fit_hgb(X: np.ndarray, y_occ: np.ndarray, y_cost: np.ndarray, seed: int) -> Dict[str, Any]:
    occ = HistGradientBoostingClassifier(
        learning_rate=HGB_LR,
        max_leaf_nodes=HGB_LEAVES,
        max_iter=HGB_ITER,
        random_state=seed,
    )
    occ.fit(X, y_occ)
    positive = y_occ == 1
    log_cost = np.log1p(y_cost[positive])
    sev = HistGradientBoostingRegressor(
        learning_rate=HGB_LR,
        max_leaf_nodes=HGB_LEAVES,
        max_iter=HGB_ITER,
        random_state=seed,
    )
    sev.fit(X[positive], log_cost)
    train_pred = sev.predict(X[positive])
    smearing = float(np.mean(np.exp(np.clip(log_cost - train_pred, -50, 50))))
    return {"occurrence_model": occ, "severity_model": sev, "smearing_factor": smearing}


def predict_glm(payload: Mapping[str, Any], X: np.ndarray) -> Dict[str, np.ndarray]:
    p = payload["occurrence_model"].predict_proba(X)[:, 1]
    mu = np.clip(payload["severity_model"].predict(X), EPS, None)
    return {"probability": p, "positive_severity": mu, "soft_cost": p * mu}


def predict_hgb(payload: Mapping[str, Any], X: np.ndarray) -> Dict[str, np.ndarray]:
    p = payload["occurrence_model"].predict_proba(X)[:, 1]
    raw = payload["severity_model"].predict(X)
    mu = np.clip(float(payload["smearing_factor"]) * np.exp(np.clip(raw, -50, 50)) - 1.0, 0.0, None)
    return {"probability": p, "positive_severity": mu, "soft_cost": p * mu}


def predict_constant(y_occ: np.ndarray, y_cost: np.ndarray, n: int) -> Dict[str, np.ndarray]:
    positive = y_occ == 1
    p = np.full(n, float(y_occ.mean()))
    mu = np.full(n, float(y_cost[positive].mean()))
    return {"probability": p, "positive_severity": mu, "soft_cost": p * mu}


def predict_persistence(test: pd.DataFrame) -> Dict[str, np.ndarray]:
    p = pd.to_numeric(test["prev_occurrence"], errors="coerce").fillna(0).to_numpy(float)
    mu = pd.to_numeric(test["cost_claims_year"], errors="coerce").fillna(0).clip(lower=0).to_numpy(float)
    return {"probability": p, "positive_severity": mu, "soft_cost": p * mu}


def select_threshold(actual: np.ndarray, probability: np.ndarray) -> float:
    best_t, best_score = 0.5, -np.inf
    for threshold in THRESHOLDS:
        score = balanced_accuracy_score(actual, (probability >= threshold).astype(int))
        if score > best_score:
            best_t, best_score = float(threshold), float(score)
    return best_t


def metrics(y_occ: np.ndarray, y_cost: np.ndarray, prediction: Mapping[str, np.ndarray], threshold: float = None) -> Dict[str, float]:
    p = np.asarray(prediction["probability"], float)
    mu = np.asarray(prediction["positive_severity"], float)
    soft = np.asarray(prediction["soft_cost"], float)
    positive = y_occ == 1
    pe = portfolio_error(y_cost, soft)
    result = {
        "roc_auc": float(roc_auc_score(y_occ, p)),
        "pr_auc": float(average_precision_score(y_occ, p)),
        "brier_score": float(brier_score_loss(y_occ, p)),
        "soft_cost_mae": float(mean_absolute_error(y_cost, soft)),
        "soft_cost_rmse": rmse(y_cost, soft),
        "soft_portfolio_error_pct": float(pe),
        "soft_abs_portfolio_error_pct": float(abs(pe)),
        "positive_severity_mae": float(mean_absolute_error(y_cost[positive], mu[positive])),
        "positive_severity_rmse": rmse(y_cost[positive], mu[positive]),
        "positive_severity_gamma_deviance": float(mean_gamma_deviance(y_cost[positive], np.clip(mu[positive], EPS, None))),
    }
    if threshold is not None:
        result["threshold"] = float(threshold)
        result["balanced_accuracy"] = float(balanced_accuracy_score(y_occ, (p >= threshold).astype(int)))
    return result


def train_baselines(paths: Mapping[str, Path], data: Mapping[str, Any], cfg: Mapping[str, Any], force: bool) -> None:
    model_root = paths["output"] / "models"
    model_root.mkdir(parents=True, exist_ok=True)
    train = data["train_idx"]
    X = data["X_dev"][train]
    y_occ = data["y_occ_dev"][train]
    y_cost = data["y_cost_dev"][train]

    glm_file = model_root / "glm_m4.joblib"
    if force or not glm_file.exists():
        print("Training M4 Logistic + Gamma GLM...", flush=True)
        joblib.dump(fit_glm(X, y_occ, y_cost), glm_file)

    for seed in cfg["seeds"]:
        model_file = model_root / "hgb" / f"seed_{seed}" / "hgb_m4.joblib"
        if force or not model_file.exists():
            model_file.parent.mkdir(parents=True, exist_ok=True)
            print(f"Training M4 HGB seed {seed}...", flush=True)
            joblib.dump(fit_hgb(X, y_occ, y_cost, int(seed)), model_file)


def load_glm(paths: Mapping[str, Path]) -> Mapping[str, Any]:
    return joblib.load(paths["output"] / "models" / "glm_m4.joblib")


def load_hgb(paths: Mapping[str, Path], seed: int) -> Mapping[str, Any]:
    return joblib.load(paths["output"] / "models" / "hgb" / f"seed_{seed}" / "hgb_m4.joblib")


def mtl_checkpoint(paths: Mapping[str, Path], seed: int) -> Path:
    return paths["m4_checkpoints"] / f"seed_{seed}" / "best_model.pt"


def policy_bootstrap(proposed: pd.DataFrame, comparator: pd.DataFrame, name: str, n_boot: int, seed: int) -> pd.DataFrame:
    keys = ["row_id", "ID_policy", "target_occurrence", "target_cost"]
    for key in keys:
        if not np.array_equal(proposed[key].to_numpy(), comparator[key].to_numpy()):
            raise RuntimeError("Prediction frames are not aligned.")

    y_occ = proposed["target_occurrence"].to_numpy(float)
    y_cost = proposed["target_cost"].to_numpy(float)
    positive = y_occ == 1
    table = proposed[keys].copy()
    table["rows_n"] = 1
    table["positive_n"] = positive.astype(int)

    for prefix, frame in [("p", proposed), ("c", comparator)]:
        prob = frame["predicted_probability"].to_numpy(float)
        sev = frame["predicted_positive_cost"].to_numpy(float)
        soft = frame["predicted_soft_cost"].to_numpy(float)
        table[f"{prefix}_sse"] = (y_cost - soft) ** 2
        table[f"{prefix}_sae"] = np.abs(y_cost - soft)
        table[f"{prefix}_brier"] = (y_occ - prob) ** 2
        table[f"{prefix}_sev_sse"] = np.where(positive, (y_cost - sev) ** 2, 0.0)

    cols = [c for c in table.columns if c not in keys]
    grouped = table.groupby("ID_policy", as_index=False)[cols].sum()
    matrix = grouped[cols].to_numpy(float)
    idx = {c: i for i, c in enumerate(cols)}
    rng = np.random.default_rng(seed)
    rows = []

    for _ in range(n_boot):
        sampled = rng.integers(0, len(grouped), size=len(grouped))
        counts = np.bincount(sampled, minlength=len(grouped)).astype(float)
        t = counts @ matrix
        n = t[idx["rows_n"]]
        n_pos = t[idx["positive_n"]]
        rows.append({
            "soft_rmse": math.sqrt(t[idx["p_sse"]] / n) - math.sqrt(t[idx["c_sse"]] / n),
            "soft_mae": t[idx["p_sae"]] / n - t[idx["c_sae"]] / n,
            "brier": t[idx["p_brier"]] / n - t[idx["c_brier"]] / n,
            "severity_rmse": math.sqrt(t[idx["p_sev_sse"]] / n_pos) - math.sqrt(t[idx["c_sev_sse"]] / n_pos),
        })

    frame = pd.DataFrame(rows)
    return pd.DataFrame([
        {
            "comparator": name,
            "metric": metric,
            "mean_difference_proposed_minus_comparator": float(frame[metric].mean()),
            "ci_2_5": float(frame[metric].quantile(0.025)),
            "ci_97_5": float(frame[metric].quantile(0.975)),
            "probability_proposed_better": float((frame[metric] < 0).mean()),
            "bootstrap_replicates": int(n_boot),
            "bootstrap_unit": "ID_policy",
        }
        for metric in frame.columns
    ])


def run_test(paths: Mapping[str, Path], data: Mapping[str, Any], cfg: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    out = paths["output"] / "test"
    tables = paths["output"] / "paper_tables"
    plots = paths["output"] / "plots"
    for folder in [out, tables, plots]:
        folder.mkdir(parents=True, exist_ok=True)

    train, val = data["train_idx"], data["val_idx"]
    X_val, X_test = data["X_dev"][val], data["X_test"]
    y_val = data["y_occ_dev"][val]
    y_occ, y_cost = data["y_occ_test"], data["y_cost_test"]
    metadata = data["test"].reset_index(drop=True)

    rows, frames = [], {}

    deterministic = {
        "Constant two-part": predict_constant(data["y_occ_dev"][train], data["y_cost_dev"][train], len(X_test)),
        "Previous-year persistence": predict_persistence(data["test"]),
    }
    for name, pred in deterministic.items():
        rows.append({"model": name, "seed": np.nan, "runs_type": "deterministic", **metrics(y_occ, y_cost, pred)})
        frames[name] = [prediction_frame(metadata, y_occ, y_cost, pred)]

    glm = load_glm(paths)
    glm_threshold = select_threshold(y_val, predict_glm(glm, X_val)["probability"])
    glm_pred = predict_glm(glm, X_test)
    rows.append({"model": "Logistic + Gamma GLM", "seed": np.nan, "runs_type": "deterministic", **metrics(y_occ, y_cost, glm_pred, glm_threshold)})
    frames["Logistic + Gamma GLM"] = [prediction_frame(metadata, y_occ, y_cost, glm_pred)]

    for seed in cfg["seeds"]:
        seed = int(seed)
        hgb = load_hgb(paths, seed)
        hgb_threshold = select_threshold(y_val, predict_hgb(hgb, X_val)["probability"])
        hgb_pred = predict_hgb(hgb, X_test)
        rows.append({"model": "HistGradientBoosting two-part", "seed": seed, "runs_type": "seeded", **metrics(y_occ, y_cost, hgb_pred, hgb_threshold)})
        frames.setdefault("HistGradientBoosting two-part", []).append(prediction_frame(metadata, y_occ, y_cost, hgb_pred))

        model, checkpoint = load_model(mtl_checkpoint(paths, seed), device)
        mtl_threshold = select_threshold(y_val, predict_model(model, checkpoint, X_val, device)["probability"])
        mtl_pred = predict_model(model, checkpoint, X_test, device)
        rows.append({"model": "M4 Shared Gamma-MTL", "seed": seed, "runs_type": "seeded", **metrics(y_occ, y_cost, mtl_pred, mtl_threshold)})
        frames.setdefault("M4 Shared Gamma-MTL", []).append(prediction_frame(metadata, y_occ, y_cost, mtl_pred))

    runs = pd.DataFrame(rows)
    runs.to_csv(out / "final_test_all_runs.csv", index=False)

    metric_cols = [
        "roc_auc", "pr_auc", "brier_score", "balanced_accuracy",
        "soft_cost_mae", "soft_cost_rmse", "soft_abs_portfolio_error_pct",
        "positive_severity_rmse", "positive_severity_gamma_deviance",
    ]
    summary_rows = []
    for name, group in runs.groupby("model", sort=False):
        row = {"model": name, "runs": int(len(group))}
        for metric_name in metric_cols:
            values = pd.to_numeric(group.get(metric_name, pd.Series(dtype=float)), errors="coerce")
            row[f"{metric_name}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric_name}_std"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else 0.0
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    order = {name: i for i, name in enumerate([
        "Constant two-part", "Previous-year persistence", "Logistic + Gamma GLM",
        "HistGradientBoosting two-part", "M4 Shared Gamma-MTL",
    ])}
    summary["_order"] = summary["model"].map(order)
    summary = summary.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    summary.to_csv(tables / "table_01_final_test_comparison_m4.csv", index=False)

    ensembles = {name: ensemble_frames(group) for name, group in frames.items()}
    for name, frame in ensembles.items():
        safe = name.lower().replace(" ", "_").replace("+", "plus").replace("-", "_")
        frame.to_csv(out / f"ensemble_{safe}.csv", index=False)

    proposed = ensembles["M4 Shared Gamma-MTL"]
    bootstrap = pd.concat([
        policy_bootstrap(proposed, ensembles[name], name, int(cfg["bootstrap"]), 20260721 + len(name))
        for name in ["Logistic + Gamma GLM", "HistGradientBoosting two-part"]
    ], ignore_index=True)
    bootstrap.to_csv(tables / "table_02_policy_bootstrap_mtl_vs_baselines.csv", index=False)

    for source, destination in [
        (paths["ablation_dev"], tables / "table_00a_m3_vs_m4_development_ablation.csv"),
        (paths["ablation_test"], tables / "table_00b_m3_vs_m4_temporal_ablation.csv"),
    ]:
        if source.exists():
            shutil.copy2(source, destination)

    plot = summary[summary["model"].isin(["Logistic + Gamma GLM", "HistGradientBoosting two-part", "M4 Shared Gamma-MTL"])].sort_values("soft_cost_rmse_mean")
    plt.figure(figsize=(8.4, 4.8))
    plt.barh(plot["model"], plot["soft_cost_rmse_mean"], xerr=plot["soft_cost_rmse_std"])
    plt.xlabel("Out-of-time soft-cost RMSE")
    plt.title("Final fair M4 comparison")
    plt.tight_layout()
    plt.savefig(plots / "final_m4_soft_cost_rmse.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(summary[["model", "roc_auc_mean", "brier_score_mean", "soft_cost_mae_mean", "soft_cost_rmse_mean", "positive_severity_rmse_mean"]].to_string(index=False))
    return {"runs": runs, "summary": summary, "ensembles": ensembles}


def count_glm(payload: Mapping[str, Any]) -> int:
    total = 0
    for model in [payload["occurrence_model"], payload["severity_model"]]:
        total += int(np.asarray(getattr(model, "coef_", [])).size)
        total += int(np.asarray(getattr(model, "intercept_", [])).size)
    return total


def count_hgb(payload: Mapping[str, Any]) -> Tuple[int, int, int]:
    predictors = nodes = leaves = 0
    for model in [payload["occurrence_model"], payload["severity_model"]]:
        for iteration in getattr(model, "_predictors", []):
            for predictor in iteration:
                predictors += 1
                node_array = getattr(predictor, "nodes", None)
                if node_array is None:
                    continue
                nodes += len(node_array)
                if "is_leaf" in (node_array.dtype.names or ()):
                    leaves += int(np.asarray(node_array["is_leaf"]).astype(bool).sum())
    return predictors, nodes, leaves


def artifact_size(payload: Any, kind: str) -> Tuple[int, float]:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / ("artifact.pt" if kind == "torch" else "artifact.joblib")
        torch.save(payload, path) if kind == "torch" else joblib.dump(payload, path, compress=0)
        size = int(path.stat().st_size)
    return size, size / 1024 ** 2


def time_function(function: Callable[[], np.ndarray], rows: int, warmup: int, repeats: int) -> Dict[str, float]:
    for _ in range(warmup):
        function()
    gc.collect()
    timings = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        output = np.asarray(function(), float)
        timings.append((time.perf_counter_ns() - start) / 1e6)
    values = np.asarray(timings)
    median = float(np.median(values))
    return {
        "median_runtime_ms_full_cohort": median,
        "mean_runtime_ms_full_cohort": float(values.mean()),
        "std_runtime_ms_full_cohort": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "median_ms_per_10000_policies": float(median * 10000 / rows),
        "throughput_policies_per_second": float(rows / (median / 1000)),
        "prediction_checksum": float(output.sum()),
    }


def run_complexity(paths: Mapping[str, Path], data: Mapping[str, Any], cfg: Mapping[str, Any], device: torch.device) -> pd.DataFrame:
    tables = paths["output"] / "paper_tables"
    plots = paths["output"] / "plots"
    tables.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["seeds"][0])
    X = data["X_test"]
    rows = len(X)

    glm = load_glm(paths)
    hgb = load_hgb(paths, seed)
    model, checkpoint = load_model(mtl_checkpoint(paths, seed), device)
    tensor = torch.from_numpy(X.astype(np.float32, copy=False)).to(device)
    scale = float(checkpoint.get("gamma_scale", checkpoint.get("gamma_cost_scale")))

    def glm_fn():
        return predict_glm(glm, X)["soft_cost"]

    def hgb_fn():
        return predict_hgb(hgb, X)["soft_cost"]

    @torch.inference_mode()
    def mtl_fn():
        logit, raw = model(tensor)
        return (torch.sigmoid(logit) * torch.exp(torch.clamp(raw, MIN_LOG_MEAN, MAX_LOG_MEAN)) * scale).cpu().numpy()

    hgb_predictors, hgb_nodes, hgb_leaves = count_hgb(hgb)
    mtl_params = count_parameters(model)
    glm_bytes, glm_mb = artifact_size(glm, "joblib")
    hgb_bytes, hgb_mb = artifact_size(hgb, "joblib")
    mtl_bytes, mtl_mb = artifact_size({"state_dict": model.state_dict(), "gamma_scale": scale}, "torch")

    specs = [
        {"model": "Logistic + Gamma GLM", "structural_measure": "fitted coefficients", "structural_count": count_glm(glm), "glm_coefficients": count_glm(glm), "tree_predictors": 0, "tree_nodes": 0, "tree_leaves": 0, "neural_parameters": 0, "serialized_size_bytes": glm_bytes, "serialized_size_mb": glm_mb, "fn": glm_fn},
        {"model": "HistGradientBoosting two-part", "structural_measure": "total tree leaves", "structural_count": hgb_leaves, "glm_coefficients": 0, "tree_predictors": hgb_predictors, "tree_nodes": hgb_nodes, "tree_leaves": hgb_leaves, "neural_parameters": 0, "serialized_size_bytes": hgb_bytes, "serialized_size_mb": hgb_mb, "fn": hgb_fn},
        {"model": "M4 Shared Gamma-MTL", "structural_measure": "trainable parameters", "structural_count": mtl_params, "glm_coefficients": 0, "tree_predictors": 0, "tree_nodes": 0, "tree_leaves": 0, "neural_parameters": mtl_params, "serialized_size_bytes": mtl_bytes, "serialized_size_mb": mtl_mb, "fn": mtl_fn},
    ]

    result_rows = []
    for spec in specs:
        timing = time_function(spec.pop("fn"), rows, int(cfg["runtime_warmup"]), int(cfg["runtime_repeats"]))
        result_rows.append({**spec, "benchmark_rows": rows, "device": "CPU", "cpu_threads": 1, "runtime_scope": "model-only expected-cost inference; preprocessing excluded", **timing})
    result = pd.DataFrame(result_rows)
    result.to_csv(tables / "table_03_complexity_m4.csv", index=False)
    return result


def encoded_groups(preprocessor: Any) -> Tuple[List[str], Dict[str, np.ndarray]]:
    encoded = list(map(str, preprocessor.get_feature_names_out()))
    raw = list(map(str, preprocessor.feature_names_in_))
    raw_sorted = sorted(raw, key=len, reverse=True)
    groups: Dict[str, List[int]] = {}
    for index, name in enumerate(encoded):
        clean = name.split("__", 1)[-1]
        if clean.startswith("missingindicator_"):
            clean = clean[len("missingindicator_"):]
        matched = next((r for r in raw_sorted if clean == r or clean.startswith(r + "_") or clean.startswith(r + "=")), clean)
        groups.setdefault(matched, []).append(index)
    ordered = sorted(groups, key=lambda feature: (min(groups[feature]), feature))
    return ordered, {feature: np.asarray(groups[feature], int) for feature in ordered}


class Predictor:
    def __init__(self, name: str, seed: int, function: Callable[[np.ndarray], Dict[str, np.ndarray]]):
        self.name, self.seed, self.function = name, int(seed), function
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        return self.function(X)


def build_predictors(paths: Mapping[str, Path], cfg: Mapping[str, Any], device: torch.device) -> List[Predictor]:
    predictors: List[Predictor] = []
    glm = load_glm(paths)
    predictors.append(Predictor("Logistic + Gamma GLM", 42, lambda X: {
        "occurrence": predict_glm(glm, X)["probability"],
        "severity": predict_glm(glm, X)["positive_severity"],
        "soft_cost": predict_glm(glm, X)["soft_cost"],
    }))

    for seed in cfg["seeds"]:
        seed = int(seed)
        hgb = load_hgb(paths, seed)
        def make_hgb(payload):
            def f(X):
                pred = predict_hgb(payload, X)
                return {"occurrence": pred["probability"], "severity": pred["positive_severity"], "soft_cost": pred["soft_cost"]}
            return f
        predictors.append(Predictor("HistGradientBoosting two-part", seed, make_hgb(hgb)))

        model, checkpoint = load_model(mtl_checkpoint(paths, seed), device)
        def make_mtl(m, c):
            def f(X):
                pred = predict_model(m, c, X.astype(np.float32, copy=False), device)
                return {"occurrence": pred["probability"], "severity": pred["positive_severity"], "soft_cost": pred["soft_cost"]}
            return f
        predictors.append(Predictor("M4 Shared Gamma-MTL", seed, make_mtl(model, checkpoint)))
    return predictors


def grouped_mc_shap(predictor: Predictor, X: np.ndarray, background: np.ndarray, groups: Sequence[str], indices: Mapping[str, np.ndarray], permutations: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    totals = {task: {group: 0.0 for group in groups} for task in TASKS}
    for permutation in range(permutations):
        current = background[rng.integers(0, len(background), size=len(X))].copy()
        previous = predictor.predict(current)
        for position in rng.permutation(len(groups)):
            group = groups[int(position)]
            idx = indices[group]
            current[:, idx] = X[:, idx]
            new = predictor.predict(current)
            for task in TASKS:
                totals[task][group] += float(np.mean(np.abs(new[task] - previous[task])))
            previous = new
        print(f"MC-SHAP | {predictor.name} | seed={predictor.seed} | {permutation + 1}/{permutations}", flush=True)

    rows = []
    for task in TASKS:
        values = np.asarray([totals[task][group] / permutations for group in groups], float)
        normalized = 100 * values / values.sum() if values.sum() else np.zeros_like(values)
        for group, value, percentage in zip(groups, values, normalized):
            rows.append({"model": predictor.name, "seed": predictor.seed, "method": "Grouped MC-SHAP", "task": task, "raw_feature": group, "importance": float(value), "normalized_importance_pct": float(percentage)})
    return pd.DataFrame(rows)


def stability(shap: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, task), group in shap.groupby(["model", "task"]):
        seeds = sorted(group["seed"].unique())
        if len(seeds) < 2:
            continue
        features = sorted(group["raw_feature"].unique())
        for a, b in combinations(seeds, 2):
            fa, fb = group[group["seed"] == a], group[group["seed"] == b]
            va = fa.set_index("raw_feature")["normalized_importance_pct"].reindex(features, fill_value=0).to_numpy()
            vb = fb.set_index("raw_feature")["normalized_importance_pct"].reindex(features, fill_value=0).to_numpy()
            top_a = set(fa.nlargest(10, "normalized_importance_pct")["raw_feature"])
            top_b = set(fb.nlargest(10, "normalized_importance_pct")["raw_feature"])
            rows.append({"model": model, "task": task, "seed_a": int(a), "seed_b": int(b), "spearman_rho": float(spearmanr(va, vb).statistic), "top10_jaccard": float(len(top_a & top_b) / len(top_a | top_b))})
    pairwise = pd.DataFrame(rows)
    if pairwise.empty:
        return pd.DataFrame()
    return pairwise.groupby(["model", "task"], as_index=False).agg(
        pair_count=("spearman_rho", "size"),
        spearman_rho_mean=("spearman_rho", "mean"),
        spearman_rho_std=("spearman_rho", "std"),
        top10_jaccard_mean=("top10_jaccard", "mean"),
        top10_jaccard_std=("top10_jaccard", "std"),
    )


def task_metric(task: str, y_occ: np.ndarray, y_cost: np.ndarray, pred: Mapping[str, np.ndarray]) -> float:
    if task == "occurrence":
        return float(roc_auc_score(y_occ, pred[task]))
    if task == "severity":
        positive = y_occ == 1
        return rmse(y_cost[positive], pred[task][positive])
    return rmse(y_cost, pred[task])


def deletion_fidelity(predictor: Predictor, shap: pd.DataFrame, X: np.ndarray, background: np.ndarray, y_occ: np.ndarray, y_cost: np.ndarray, groups: Sequence[str], indices: Mapping[str, np.ndarray], repeats: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    original = predictor.predict(X)
    rows = []
    for task in TASKS:
        importance = shap[(shap["model"] == predictor.name) & (shap["seed"] == predictor.seed) & (shap["task"] == task)].sort_values("normalized_importance_pct", ascending=False)
        top = list(importance.head(5)["raw_feature"])
        remaining = [group for group in groups if group not in top]
        baseline_metric = task_metric(task, y_occ, y_cost, original)
        for repeat in range(repeats):
            donors = background[rng.integers(0, len(background), size=len(X))]
            top_X, random_X = X.copy(), X.copy()
            for feature in top:
                top_X[:, indices[feature]] = donors[:, indices[feature]]
            random_features = list(rng.choice(remaining, size=min(5, len(remaining)), replace=False))
            for feature in random_features:
                random_X[:, indices[feature]] = donors[:, indices[feature]]
            top_metric = task_metric(task, y_occ, y_cost, predictor.predict(top_X))
            random_metric = task_metric(task, y_occ, y_cost, predictor.predict(random_X))
            if task == "occurrence":
                top_d, random_d, metric_name = baseline_metric - top_metric, baseline_metric - random_metric, "ROC-AUC drop"
            else:
                top_d, random_d, metric_name = top_metric - baseline_metric, random_metric - baseline_metric, "RMSE increase"
            rows.append({"model": predictor.name, "seed": predictor.seed, "task": task, "repeat": repeat + 1, "fidelity_metric": metric_name, "top5_deterioration": float(top_d), "random5_deterioration": float(random_d), "fidelity_gain": float(top_d - random_d), "top_features": "|".join(top)})
    return pd.DataFrame(rows)


def integrated_gradients(model: SharedGammaMTL, checkpoint: Mapping[str, Any], X: np.ndarray, groups: Sequence[str], indices: Mapping[str, np.ndarray], steps: int, batch_size: int, device: torch.device) -> pd.DataFrame:
    model.eval()
    scale = float(checkpoint.get("gamma_scale", checkpoint.get("gamma_cost_scale")))
    totals = {task: np.zeros(X.shape[1], float) for task in TASKS}
    rows_seen = 0
    for start in range(0, len(X), batch_size):
        batch = torch.from_numpy(X[start:start + batch_size].astype(np.float32, copy=False)).to(device)
        base = torch.zeros_like(batch)
        delta = batch - base
        for task in TASKS:
            grad_sum = torch.zeros_like(batch)
            for step in range(1, steps + 1):
                z = (base + (step / steps) * delta).requires_grad_(True)
                logit, raw = model(z)
                p = torch.sigmoid(logit)
                mu = torch.exp(torch.clamp(raw, MIN_LOG_MEAN, MAX_LOG_MEAN)) * scale
                out = p if task == "occurrence" else mu if task == "severity" else p * mu
                grad_sum += torch.autograd.grad(out.sum(), z)[0].detach()
            totals[task] += (delta * grad_sum / steps).abs().sum(dim=0).cpu().numpy()
        rows_seen += len(batch)
        print(f"IG | {rows_seen}/{len(X)} rows", flush=True)

    rows = []
    for task in TASKS:
        encoded = totals[task] / rows_seen
        values = np.asarray([encoded[indices[group]].sum() for group in groups], float)
        normalized = 100 * values / values.sum() if values.sum() else np.zeros_like(values)
        for group, value, percentage in zip(groups, values, normalized):
            rows.append({"method": "Integrated Gradients", "task": task, "raw_feature": group, "importance": float(value), "normalized_importance_pct": float(percentage)})
    return pd.DataFrame(rows)


def run_explainability(paths: Mapping[str, Path], data: Mapping[str, Any], cfg: Mapping[str, Any], device: torch.device, force: bool) -> None:
    out = paths["output"] / "explainability"
    runs = out / "runs"
    tables = paths["output"] / "paper_tables"
    plots = paths["output"] / "plots"
    for folder in [out, runs, tables, plots]:
        folder.mkdir(parents=True, exist_ok=True)

    groups, indices = encoded_groups(data["preprocessor"])
    rng = np.random.default_rng(20260721)
    X_train = data["X_dev"][data["train_idx"]]
    X_test = data["X_test"]
    explain_idx = rng.choice(len(X_test), size=min(int(cfg["shap_rows"]), len(X_test)), replace=False)
    background_idx = rng.choice(len(X_train), size=min(int(cfg["background_rows"]), len(X_train)), replace=False)
    X_explain = X_test[explain_idx]
    background = X_train[background_idx]

    predictors = build_predictors(paths, cfg, device)
    shap_parts = []
    for predictor in predictors:
        safe = predictor.name.lower().replace(" ", "_").replace("+", "plus").replace("-", "_")
        file = runs / f"{safe}_seed_{predictor.seed}_mc_shap.csv"
        if file.exists() and not force:
            frame = pd.read_csv(file)
        else:
            frame = grouped_mc_shap(predictor, X_explain, background, groups, indices, int(cfg["shap_permutations"]), 1000 + predictor.seed + len(predictor.name))
            frame.to_csv(file, index=False)
        shap_parts.append(frame)

    shap = pd.concat(shap_parts, ignore_index=True)
    shap.to_csv(out / "grouped_mc_shap_all_runs.csv", index=False)
    shap_summary = shap.groupby(["model", "task", "raw_feature"], as_index=False).agg(
        importance_mean=("importance", "mean"),
        importance_std=("importance", "std"),
        normalized_importance_pct_mean=("normalized_importance_pct", "mean"),
        normalized_importance_pct_std=("normalized_importance_pct", "std"),
        seeds=("seed", "nunique"),
    )
    shap_summary.to_csv(out / "grouped_mc_shap_summary.csv", index=False)

    stability_table = stability(shap)
    stability_table.to_csv(out / "shap_stability_summary.csv", index=False)

    fidelity_idx = rng.choice(len(X_test), size=min(int(cfg["fidelity_rows"]), len(X_test)), replace=False)
    fidelity_parts = [
        deletion_fidelity(
            predictor,
            shap,
            X_test[fidelity_idx],
            background,
            data["y_occ_test"][fidelity_idx],
            data["y_cost_test"][fidelity_idx],
            groups,
            indices,
            int(cfg["fidelity_repeats"]),
            3000 + predictor.seed + len(predictor.name),
        )
        for predictor in predictors
    ]
    fidelity = pd.concat(fidelity_parts, ignore_index=True)
    fidelity.to_csv(out / "deletion_fidelity_all_runs.csv", index=False)
    fidelity_summary = fidelity.groupby(["model", "task", "fidelity_metric"], as_index=False).agg(
        top5_deterioration_mean=("top5_deterioration", "mean"),
        top5_deterioration_std=("top5_deterioration", "std"),
        random5_deterioration_mean=("random5_deterioration", "mean"),
        random5_deterioration_std=("random5_deterioration", "std"),
        fidelity_gain_mean=("fidelity_gain", "mean"),
        fidelity_gain_std=("fidelity_gain", "std"),
        runs=("fidelity_gain", "size"),
    )
    fidelity_summary.merge(stability_table, on=["model", "task"], how="left").to_csv(tables / "table_04_explainability_stability_fidelity.csv", index=False)

    top_features = shap_summary.sort_values(["model", "task", "normalized_importance_pct_mean"], ascending=[True, True, False]).groupby(["model", "task"], group_keys=False).head(10)
    top_features.to_csv(tables / "table_05_top_features_by_model_task.csv", index=False)

    ig_idx = rng.choice(len(X_test), size=min(int(cfg["ig_rows"]), len(X_test)), replace=False)
    ig_parts = []
    for seed in cfg["seeds"]:
        seed = int(seed)
        file = runs / f"m4_gamma_mtl_seed_{seed}_ig.csv"
        if file.exists() and not force:
            frame = pd.read_csv(file)
        else:
            model, checkpoint = load_model(mtl_checkpoint(paths, seed), device)
            frame = integrated_gradients(model, checkpoint, X_test[ig_idx], groups, indices, int(cfg["ig_steps"]), int(cfg["ig_batch"]), device)
            frame["model"] = "M4 Shared Gamma-MTL"
            frame["seed"] = seed
            frame.to_csv(file, index=False)
        ig_parts.append(frame)
    ig = pd.concat(ig_parts, ignore_index=True)
    ig.to_csv(out / "integrated_gradients_all_runs.csv", index=False)

    agreement_rows = []
    for seed in cfg["seeds"]:
        for task in TASKS:
            sf = shap[(shap["model"] == "M4 Shared Gamma-MTL") & (shap["seed"] == int(seed)) & (shap["task"] == task)]
            igf = ig[(ig["seed"] == int(seed)) & (ig["task"] == task)]
            features = sorted(set(sf["raw_feature"]) | set(igf["raw_feature"]))
            sv = sf.set_index("raw_feature")["normalized_importance_pct"].reindex(features, fill_value=0).to_numpy()
            iv = igf.set_index("raw_feature")["normalized_importance_pct"].reindex(features, fill_value=0).to_numpy()
            top_s = set(sf.nlargest(10, "normalized_importance_pct")["raw_feature"])
            top_i = set(igf.nlargest(10, "normalized_importance_pct")["raw_feature"])
            agreement_rows.append({"seed": int(seed), "task": task, "spearman_rho": float(spearmanr(sv, iv).statistic), "top10_jaccard": float(len(top_s & top_i) / len(top_s | top_i))})
    agreement = pd.DataFrame(agreement_rows)
    agreement.groupby("task", as_index=False).agg(
        spearman_rho_mean=("spearman_rho", "mean"),
        spearman_rho_std=("spearman_rho", "std"),
        top10_jaccard_mean=("top10_jaccard", "mean"),
        top10_jaccard_std=("top10_jaccard", "std"),
        seeds=("seed", "nunique"),
    ).to_csv(tables / "table_06_mtl_ig_shap_agreement.csv", index=False)

    proposed = shap_summary[shap_summary["model"] == "M4 Shared Gamma-MTL"]
    occurrence = proposed[proposed["task"] == "occurrence"].set_index("raw_feature")["normalized_importance_pct_mean"]
    severity = proposed[proposed["task"] == "severity"].set_index("raw_feature")["normalized_importance_pct_mean"]
    features = sorted(set(occurrence.index) | set(severity.index))
    contrast = pd.DataFrame({"raw_feature": features, "occurrence_minus_severity": [float(occurrence.get(f, 0) - severity.get(f, 0)) for f in features]})
    contrast["absolute"] = contrast["occurrence_minus_severity"].abs()
    contrast = contrast.nlargest(12, "absolute").sort_values("occurrence_minus_severity")
    contrast.to_csv(out / "m4_mtl_occurrence_severity_contrast.csv", index=False)
    plt.figure(figsize=(9, 6))
    plt.barh(contrast["raw_feature"], contrast["occurrence_minus_severity"])
    plt.axvline(0, linewidth=1)
    plt.xlabel("Normalized SHAP importance: occurrence − severity (pp)")
    plt.title("M4 Gamma-MTL task-specific feature contrast")
    plt.tight_layout()
    plt.savefig(plots / "m4_mtl_task_contrast.png", dpi=300, bbox_inches="tight")
    plt.close()


def write_protocol(paths: Mapping[str, Path], data: Mapping[str, Any], args: argparse.Namespace, cfg: Mapping[str, Any]) -> None:
    save_json({
        "study": "Final fair M4 model comparison and explainability",
        "same_inputs_for_learned_models": True,
        "encoded_input_dim": int(data["X_dev"].shape[1]),
        "models": ["Constant two-part", "Previous-year persistence", "Logistic + Gamma GLM", "HistGradientBoosting two-part", "M4 Shared Gamma-MTL"],
        "seeds": list(cfg["seeds"]),
        "glm_C": GLM_C,
        "gamma_alpha": GAMMA_ALPHA,
        "hgb_learning_rate": HGB_LR,
        "hgb_max_leaf_nodes": HGB_LEAVES,
        "hgb_max_iter": HGB_ITER,
        "reporting_boundary": "2018->2019 was already inspected; frozen post-hoc out-of-time comparison.",
        "stage": args.stage,
        "profile": args.profile,
    }, paths["output"] / "protocol_and_integrity.json")


def write_readme(paths: Mapping[str, Path]) -> None:
    (paths["output"] / "README_RESULTS.txt").write_text(
        """FINAL M4 PAPER OUTPUTS
======================

Focused feature ablation:
- table_00a_m3_vs_m4_development_ablation.csv
- table_00b_m3_vs_m4_temporal_ablation.csv

Final fair M4 comparison:
- table_01_final_test_comparison_m4.csv
- table_02_policy_bootstrap_mtl_vs_baselines.csv
- table_03_complexity_m4.csv

Final explainability:
- table_04_explainability_stability_fidelity.csv
- table_05_top_features_by_model_task.csv
- table_06_mtl_ig_shap_agreement.csv

All learned models use the same frozen M4 representation and development split.
The 2018->2019 cohort is a frozen post-hoc out-of-time comparison.
""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    cfg = PROFILES[args.profile]
    paths = resolve_paths(args.project_dir, args.profile)
    paths["output"].mkdir(parents=True, exist_ok=True)
    device = configure_cpu()
    set_seed(int(cfg["seeds"][0]))

    print("=" * 100)
    print("FINAL FAIR M4 COMPARISON + EXPLAINABILITY")
    print("=" * 100)
    print(f"Project: {paths['project']}")
    print(f"Profile: {args.profile}")
    print(f"Stage:   {args.stage}")
    print(f"Seeds:   {cfg['seeds']}")

    data = prepare_data(paths)
    write_protocol(paths, data, args, cfg)

    stages = ["train", "test", "complexity", "explainability"] if args.stage == "all" else [args.stage]
    if "train" in stages:
        train_baselines(paths, data, cfg, args.force)
    if "test" in stages:
        run_test(paths, data, cfg, device)
    if "complexity" in stages:
        run_complexity(paths, data, cfg, device)
    if "explainability" in stages:
        run_explainability(paths, data, cfg, device, args.force)

    write_readme(paths)
    print("=" * 100)
    print("COMPLETE")
    print(f"Outputs: {paths['output']}")
    print("Send paper_tables/, plots/, protocol_and_integrity.json and the log.")


if __name__ == "__main__":
    main()
