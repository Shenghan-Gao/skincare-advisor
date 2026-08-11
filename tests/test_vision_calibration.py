import numpy as np
import pytest
import torch

from scripts.calibrate_vision_concerns import ensure_validation_split
from skincare.config import CONCERNS
from skincare.vision.calibration import (
    bootstrap_concern_macro_f1,
    concern_metrics,
    search_concern_thresholds,
    shift_logits_to_threshold,
)


def _example():
    truth = np.array([0, 0, 1, 1], dtype=float)
    probability = np.array([0.1, 0.55, 0.65, 0.9])
    return (
        np.column_stack([truth] * len(CONCERNS)),
        np.column_stack([probability] * len(CONCERNS)),
    )


def test_threshold_search_improves_macro_f1_and_is_per_class():
    targets, probabilities = _example()
    default = concern_metrics(targets, probabilities, [0.5] * len(CONCERNS))
    thresholds = search_concern_thresholds(
        targets, probabilities, grid=np.array([0.5, 0.6, 0.7])
    )
    calibrated = concern_metrics(targets, probabilities, thresholds)

    assert thresholds.tolist() == pytest.approx([0.6] * len(CONCERNS))
    assert calibrated["macro_f1"] == 1.0
    assert calibrated["macro_f1"] > default["macro_f1"]


def test_bootstrap_interval_is_deterministic_and_bounded():
    targets, probabilities = _example()
    thresholds = np.full(len(CONCERNS), 0.6)
    first = bootstrap_concern_macro_f1(
        targets, probabilities, thresholds, samples=100, seed=7
    )
    second = bootstrap_concern_macro_f1(
        targets, probabilities, thresholds, samples=100, seed=7
    )

    assert first == second
    assert 0 <= first["lower"] <= first["point_estimate"] <= first["upper"] <= 1


def test_threshold_logit_shift_preserves_shape_and_maps_cutoff_to_half():
    thresholds = torch.tensor([0.2, 0.3, 0.4, 0.6, 0.7, 0.8])
    logits = torch.logit(thresholds).unsqueeze(0)

    shifted = shift_logits_to_threshold(logits, thresholds)

    assert shifted.shape == logits.shape
    assert torch.sigmoid(shifted).tolist()[0] == pytest.approx([0.5] * len(CONCERNS))


def test_threshold_search_rejects_test_split():
    ensure_validation_split("data/processed/vision_val.csv")
    with pytest.raises(ValueError, match="never the one-shot test"):
        ensure_validation_split("data/processed/vision_test.csv")
