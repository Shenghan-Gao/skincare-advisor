import json

import pandas as pd

from skincare.augment.diffusion_aug import build_augmented_csv, build_oversampled_csv
from skincare.eval.rag_eval import evaluate_rankings, groundedness_rate, precision_at_k


def _vision_df():
    return pd.DataFrame(
        [
            {
                "filepath": "a.jpg",
                "skin_type": "combination",
                "acne": None,
                "dark_spots": None,
                "redness": None,
                "large_pores": None,
                "wrinkles": None,
                "dryness": None,
            },
            {
                "filepath": "b.jpg",
                "skin_type": "oily",
                "acne": 1,
                "dark_spots": 0,
                "redness": None,
                "large_pores": 0,
                "wrinkles": 0,
                "dryness": 0,
            },
            {
                "filepath": "c.jpg",
                "skin_type": "combination",
                "acne": None,
                "dark_spots": None,
                "redness": None,
                "large_pores": None,
                "wrinkles": None,
                "dryness": None,
            },
        ]
    )


def test_oversampling_preserves_schema_and_unknowns(tmp_path):
    orig = _vision_df()
    orig_path = tmp_path / "vision_train.csv"
    out_path = tmp_path / "vision_train_aug.csv"
    orig.to_csv(orig_path, index=False)

    merged = build_oversampled_csv(str(orig_path), "combination", 2, str(out_path), seed=42)

    assert list(merged.columns) == list(orig.columns)
    assert len(merged) == 5
    assert (merged["skin_type"] == "combination").sum() == 4
    assert merged.tail(2)["acne"].isna().all()


def test_build_augmented_csv_only_uses_human_accepted_rows(tmp_path):
    orig = _vision_df()
    orig_path = tmp_path / "vision_train.csv"
    orig.to_csv(orig_path, index=False)
    synth_dir = tmp_path / "synth"
    synth_dir.mkdir()
    pd.DataFrame(
        [
            {
                "filepath": "synthetic/accepted.jpg",
                "source_filepath": "b.jpg",
                "accepted": 1,
            },
            {
                "filepath": "synthetic/rejected.jpg",
                "source_filepath": "a.jpg",
                "accepted": 0,
            },
        ]
    ).to_csv(synth_dir / "metadata.csv", index=False)

    out_path = tmp_path / "vision_train_aug.csv"
    merged = build_augmented_csv(str(orig_path), synth_dir, str(out_path))

    assert len(merged) == len(orig) + 1
    added = merged.iloc[-1]
    assert added["filepath"] == "synthetic/accepted.jpg"
    assert added["skin_type"] == "oily"
    assert added["acne"] == 1
    assert pd.isna(added["redness"])
    assert "synthetic/rejected.jpg" not in set(merged["filepath"])


def test_build_augmented_csv_with_no_accepted_rows_keeps_original(tmp_path):
    orig = _vision_df()
    orig_path = tmp_path / "vision_train.csv"
    orig.to_csv(orig_path, index=False)
    synth_dir = tmp_path / "synth"
    synth_dir.mkdir()
    pd.DataFrame(
        [
            {
                "filepath": "synthetic/rejected.jpg",
                "source_filepath": "a.jpg",
                "accepted": 0,
            }
        ]
    ).to_csv(synth_dir / "metadata.csv", index=False)

    out_path = tmp_path / "vision_train_aug.csv"
    merged = build_augmented_csv(str(orig_path), synth_dir, str(out_path))
    assert len(merged) == len(orig)
    pd.testing.assert_frame_equal(merged, orig, check_dtype=False)


def test_rag_precision_and_evaluation(tmp_path):
    labels = pd.DataFrame(
        [
            {"query_id": "Q1", "product_id": "P1", "relevant": 1},
            {"query_id": "Q1", "product_id": "P2", "relevant": 0},
            {"query_id": "Q1", "product_id": "P3", "relevant": 1},
            {"query_id": "Q2", "product_id": "P4", "relevant": 1},
            {"query_id": "Q2", "product_id": "P5", "relevant": 1},
            {"query_id": "Q2", "product_id": "P6", "relevant": 0},
        ]
    )
    labels_path = tmp_path / "labels.csv"
    labels.to_csv(labels_path, index=False)
    rankings = {
        "systems": {
            "model_a": {
                "per_query": [
                    {"query_id": "Q1", "ranked_product_ids": ["P1", "P2", "P3"]},
                    {"query_id": "Q2", "ranked_product_ids": ["P4", "P5", "P6"]},
                ]
            }
        }
    }
    rankings_path = tmp_path / "rankings.json"
    rankings_path.write_text(json.dumps(rankings), encoding="utf-8")

    result = evaluate_rankings(labels_path, rankings_path, k=3)
    assert result["systems"]["model_a"]["precision_at_3"] == 2 / 3
    assert precision_at_k({"P1": 1, "P2": 0, "P3": 1}, ["P1", "P2", "P3"], 3) == 2 / 3
    assert groundedness_rate(["E1", "E2"], ["E2", "E3"]) == 0.5
