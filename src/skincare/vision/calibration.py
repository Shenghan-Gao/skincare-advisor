"""Concern threshold search and uncertainty estimates for sparse labels."""

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support

from skincare.config import CONCERNS


def concern_metrics(targets, probabilities, thresholds):
    """Score each concern only where ground truth is available."""
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.shape != (len(CONCERNS),):
        raise ValueError(f"expected {len(CONCERNS)} thresholds, got {thresholds.shape}")
    predictions = probabilities >= thresholds.reshape(1, -1)
    per_class = {}
    for index, name in enumerate(CONCERNS):
        column = targets[:, index]
        valid = np.isfinite(column) & (column >= 0)
        if not valid.any():
            per_class[name] = {
                "precision": None, "recall": None, "f1": None,
                "support": 0, "positive_support": 0,
                "threshold": float(thresholds[index]),
            }
            continue
        truth = column[valid].astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            truth, predictions[valid, index], average="binary", zero_division=0,
        )
        per_class[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(valid.sum()),
            "positive_support": int(truth.sum()),
            "threshold": float(thresholds[index]),
        }
    scores = [value["f1"] for value in per_class.values() if value["f1"] is not None]
    return {
        "macro_f1": float(np.mean(scores)) if scores else 0.0,
        "per_class": per_class,
    }


def search_concern_thresholds(targets, probabilities, grid=None):
    """Choose one validation threshold per concern, breaking ties toward 0.5."""
    if grid is None:
        grid = np.linspace(0.05, 0.95, 91)
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or not grid.size or np.any((grid <= 0) | (grid >= 1)):
        raise ValueError("threshold grid must be a non-empty 1-D array inside (0, 1)")
    thresholds = []
    for index in range(len(CONCERNS)):
        column = targets[:, index]
        valid = np.isfinite(column) & (column >= 0)
        if not valid.any():
            thresholds.append(0.5)
            continue
        truth = column[valid].astype(int)
        scores = []
        for threshold in grid:
            prediction = probabilities[valid, index] >= threshold
            _, _, f1, _ = precision_recall_fscore_support(
                truth, prediction, average="binary", zero_division=0,
            )
            scores.append(float(f1))
        best_score = max(scores)
        candidates = [
            float(threshold) for threshold, score in zip(grid, scores)
            if np.isclose(score, best_score)
        ]
        thresholds.append(min(candidates, key=lambda value: (abs(value - 0.5), value)))
    return np.asarray(thresholds, dtype=float)


def bootstrap_concern_macro_f1(
    targets,
    probabilities,
    thresholds,
    samples=1000,
    confidence=0.95,
    seed=42,
):
    """Row-bootstrap labelled validation examples with fixed chosen thresholds."""
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    valid_rows = np.flatnonzero(
        (np.isfinite(targets) & (targets >= 0)).any(axis=1)
    )
    if not valid_rows.size:
        raise ValueError("cannot bootstrap without labelled concern rows")
    rng = np.random.default_rng(seed)
    scores = np.empty(samples, dtype=float)
    for index in range(samples):
        chosen = rng.choice(valid_rows, size=len(valid_rows), replace=True)
        scores[index] = concern_metrics(
            targets[chosen], probabilities[chosen], thresholds
        )["macro_f1"]
    alpha = (1 - confidence) / 2
    point = concern_metrics(targets, probabilities, thresholds)["macro_f1"]
    return {
        "point_estimate": point,
        "lower": float(np.quantile(scores, alpha)),
        "upper": float(np.quantile(scores, 1 - alpha)),
        "confidence": confidence,
        "bootstrap_samples": samples,
        "seed": seed,
        "resampled_labelled_rows": len(valid_rows),
        "thresholds_fixed_during_bootstrap": True,
        "uncertainty_scope": (
            "row-sampling uncertainty conditional on fixed thresholds; "
            "does not include threshold-selection bias"
        ),
    }


def shift_logits_to_threshold(logits, thresholds):
    """Map each raw decision threshold to 0.5 without changing output shape."""
    threshold_tensor = torch.as_tensor(
        thresholds, dtype=logits.dtype, device=logits.device
    ).clamp(1e-6, 1 - 1e-6)
    if threshold_tensor.shape != (len(CONCERNS),):
        raise ValueError(
            f"expected {len(CONCERNS)} thresholds, got {tuple(threshold_tensor.shape)}"
        )
    return logits - torch.logit(threshold_tensor)
