import pytest

from skincare.rag.score_human_eval import score_mapping


def test_score_mapping_computes_precision_at_three():
    mapping = {
        "top_k": 3,
        "systems": {
            "Q001": {"a": ["P1", "P2", "P3"], "b": ["P1", "P4", "P5"]},
            "Q002": {"a": ["P4", "P5", "P6"], "b": ["P2", "P3", "P6"]},
        },
    }
    labels = {
        ("Q001", "P1"): 1, ("Q001", "P2"): 1, ("Q001", "P3"): 0,
        ("Q001", "P4"): 0, ("Q001", "P5"): 0,
        ("Q002", "P2"): 1, ("Q002", "P3"): 0, ("Q002", "P4"): 1,
        ("Q002", "P5"): 0, ("Q002", "P6"): 1,
    }
    result = score_mapping(mapping, labels)
    assert result["systems"]["a"]["precision_at_3"] == pytest.approx(2 / 3)
    assert result["systems"]["b"]["precision_at_3"] == pytest.approx(0.5)


def test_score_mapping_rejects_missing_label():
    mapping = {"top_k": 3, "systems": {"Q001": {"a": ["P1", "P2", "P3"]}}}
    with pytest.raises(ValueError, match="missing label"):
        score_mapping(mapping, {("Q001", "P1"): 1})
