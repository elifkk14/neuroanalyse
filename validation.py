from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bins: list[float] | tuple[float, ...],
    auc_thresholds: list[float] | tuple[float, ...],
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    metrics: dict[str, float] = {
        "MAE_centiloid_raw": float(abs_errors.mean()),
        "RMSE_centiloid_raw": float(np.sqrt(np.mean(errors**2))),
        "Pearson_r": pearson_r(y_true, y_pred),
        "Spearman_r": spearman_r(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }

    metrics.update(mae_by_centiloid_bin(y_true, abs_errors, bins))
    for threshold in auc_thresholds:
        auc = binary_auc((y_true > threshold).astype(np.int32), y_pred)
        metrics[f"AUC_centiloid_gt_{threshold:g}"] = auc

    return metrics


def mae_by_centiloid_bin(
    y_true: np.ndarray,
    abs_errors: np.ndarray,
    bins: list[float] | tuple[float, ...],
) -> dict[str, float]:
    output: dict[str, float] = {}
    bins_array = np.asarray(bins, dtype=np.float64)
    bin_indices = np.digitize(y_true, bins_array, right=False) - 1

    for index in range(len(bins_array) - 1):
        low = bins_array[index]
        high = bins_array[index + 1]
        mask = bin_indices == index
        key = f"MAE_bin_{low:g}_to_{high:g}"
        output[key] = float(abs_errors[mask].mean()) if np.any(mask) else float("nan")

    return output


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def spearman_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_rank = pd.Series(y_true).rank(method="average").to_numpy()
    pred_rank = pd.Series(y_pred).rank(method="average").to_numpy()
    return pearson_r(true_rank, pred_rank)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    total = np.sum((y_true - y_true.mean()) ** 2)
    if total == 0:
        return float("nan")
    residual = np.sum((y_true - y_pred) ** 2)
    return float(1.0 - residual / total)


def binary_auc(y_true_binary: np.ndarray, y_score: np.ndarray) -> float:
    y_true_binary = np.asarray(y_true_binary, dtype=np.int32).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    positives = int(y_true_binary.sum())
    negatives = int(len(y_true_binary) - positives)
    if positives == 0 or negatives == 0:
        return float("nan")

    ranks = pd.Series(y_score).rank(method="average").to_numpy()
    positive_rank_sum = ranks[y_true_binary == 1].sum()
    auc = (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc)


def clean_metric_dict(metrics: dict[str, Any]) -> dict[str, float]:
    cleaned: dict[str, float] = {}
    for key, value in metrics.items():
        value = float(value)
        if np.isfinite(value):
            cleaned[key] = value
    return cleaned
