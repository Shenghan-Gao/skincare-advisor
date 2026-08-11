import numpy as np
import pytest

from scripts.evaluate_vision_checkpoints import (
    baseline_rows,
    concern_correlation,
    ensure_split_matches_csv,
)


def test_baselines_are_computed_from_the_supplied_labels():
    type_true = np.array([0, 0, 0, 1, 2, 3])
    concern_true = np.array(
        [
            [1, 0, 1, 0, 1, 0],
            [0, 1, 1, 0, 1, 0],
            [1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 1],
        ],
        dtype=float,
    )

    type_baseline, concern_baseline = baseline_rows(type_true, concern_true)

    assert type_baseline["type_accuracy"] == 0.5
    assert type_baseline["concern_macro_f1"] is None
    assert concern_baseline["type_accuracy"] is None
    assert 0 <= concern_baseline["concern_macro_f1"] <= 1


def test_concern_correlation_counts_highly_correlated_pairs():
    base = np.linspace(0.1, 0.9, 20)
    probabilities = np.column_stack([base * (index + 1) / 6 for index in range(6)])

    result = concern_correlation(probabilities)

    assert result["off_diagonal"]["total_pairs"] == 15
    assert result["off_diagonal"]["pairs_at_least_0_8"] == 15
    assert result["off_diagonal"]["max"] <= 1.0


def test_evaluation_split_must_match_csv_name():
    ensure_split_matches_csv("data/processed/vision_val.csv", "validation")
    ensure_split_matches_csv("data/processed/vision_test.csv", "test")

    with pytest.raises(ValueError, match="split mismatch"):
        ensure_split_matches_csv("data/processed/vision_val.csv", "test")
    with pytest.raises(ValueError, match="split mismatch"):
        ensure_split_matches_csv("data/processed/vision_test.csv", "validation")
