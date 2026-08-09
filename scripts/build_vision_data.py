"""Member A: clean and split the facial skin image datasets.

Main source expected at:
  data/raw/vision/skin_type_classification_dataset/
    train/{oily,dry,combination,normal}/*.jpg
    valid/{...}/*.jpg
    test/{...}/*.jpg
    skinalaysis_labeling_train1.xlsx
    skinanalysis_valid1.xlsx

Optional auxiliary concern source expected at:
  data/raw/vision/skin_defects/
    files/acne/... , files/redness/... , files/bags/...
    skin_defects.csv

Outputs:
  data/processed/vision_{train,val,test}.csv
  data/processed/class_distribution.md
  data/processed/vision_concern_aux.csv
  data/processed/vision_final_audit.json

Important data-integrity choice:
The 4,093-image skin-type source only provides detailed concern annotations for
200 spreadsheet rows.  Missing concern labels stay blank (NaN); they are NOT
silently converted to 0.  Teammate B must mask missing concern targets in the
multi-task loss (see docs/MEMBER_A_HANDOFF.md).
"""
from __future__ import annotations

import json
import math
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
VISION = ROOT / "data" / "raw" / "vision"
MAIN = VISION / "skin_type_classification_dataset"
DEFECTS = VISION / "skin_defects"
OUT = ROOT / "data" / "processed"
CONCERNS = ["acne", "dark_spots", "redness", "large_pores", "wrinkles", "dryness"]
SKIN_TYPES = ["oily", "dry", "combination", "normal"]
SEED = 5560


def source_id(path_or_name: str | Path) -> str:
    """Recover the original source-image id from Roboflow-style filenames."""
    name = os.path.basename(str(path_or_name))
    name = re.sub(
        r"_(?:jpg|jpeg|png)\.rf\.[0-9a-f]+(?:\.(?:jpg|jpeg|png))?$",
        "",
        name,
        flags=re.I,
    )
    name = re.sub(r"^(oily|dry|normal|combination)_", "", name, flags=re.I)
    name = re.sub(r"^(?:new[_-]*|[-_]+)+", "", name, flags=re.I)
    name = re.sub(r"-copy$", "", name, flags=re.I)
    return name.lower()


def _dct_matrix(n: int = 32) -> np.ndarray:
    """Orthonormal DCT-II matrix, avoiding an extra scipy/opencv dependency."""
    k = np.arange(n).reshape(-1, 1)
    i = np.arange(n).reshape(1, -1)
    mat = np.cos(np.pi * (i + 0.5) * k / n)
    mat[0, :] *= math.sqrt(1 / n)
    mat[1:, :] *= math.sqrt(2 / n)
    return mat.astype(np.float32)


_DCT = _dct_matrix(32)


def phash64(path: Path) -> int:
    """64-bit perceptual hash using the standard 32x32 DCT / 8x8 low-frequency block."""
    with Image.open(path) as im:
        arr = np.asarray(im.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    dct = _DCT @ arr @ _DCT.T
    low = dct[:8, :8].ravel()
    median = np.median(low[1:])
    value = 0
    for bit in low > median:
        value = (value << 1) | int(bool(bit))
    return value


def audit_and_dedupe() -> tuple[list[Path], dict]:
    files = sorted(p for p in MAIN.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    valid: list[Path] = []
    invalid: list[dict] = []

    for path in files:
        try:
            with Image.open(path) as im:
                width, height, mode = im.size[0], im.size[1], im.mode
                im.verify()
            if width < 100 or height < 100 or mode != "RGB":
                invalid.append({"filepath": str(path), "reason": f"{width}x{height}, mode={mode}"})
            else:
                valid.append(path)
        except Exception as exc:  # pragma: no cover - depends on corrupt input files
            invalid.append({"filepath": str(path), "reason": f"open_error: {exc}"})

    # Greedy pHash dedupe within each class.  Threshold is the team's frozen spec.
    kept_by_class: dict[str, list[tuple[Path, int]]] = defaultdict(list)
    within_removed: list[dict] = []
    for path in valid:
        skin_type = path.parent.name
        h = phash64(path)
        duplicate = None
        for kept_path, kept_hash in kept_by_class[skin_type]:
            distance = (h ^ kept_hash).bit_count()
            if distance <= 5:
                duplicate = (kept_path, distance)
                break
        if duplicate:
            within_removed.append(
                {
                    "filepath": str(path),
                    "duplicate_of": str(duplicate[0]),
                    "distance": duplicate[1],
                    "skin_type": skin_type,
                }
            )
        else:
            kept_by_class[skin_type].append((path, h))

    # If an almost-identical image appears under two different skin labels, the
    # label itself is ambiguous.  Remove BOTH images instead of choosing a label.
    cross_pairs: list[dict] = []
    cross_paths: set[Path] = set()
    classes = sorted(kept_by_class)
    for i, class_a in enumerate(classes):
        for class_b in classes[i + 1 :]:
            for path_a, hash_a in kept_by_class[class_a]:
                for path_b, hash_b in kept_by_class[class_b]:
                    distance = (hash_a ^ hash_b).bit_count()
                    if distance <= 5:
                        cross_paths.update({path_a, path_b})
                        cross_pairs.append(
                            {
                                "a": str(path_a),
                                "class_a": class_a,
                                "b": str(path_b),
                                "class_b": class_b,
                                "distance": distance,
                            }
                        )

    final = [
        path
        for values in kept_by_class.values()
        for path, _ in values
        if path not in cross_paths
    ]

    # The vendor-provided split leaks augmented copies of the same source image.
    original_group_splits: dict[str, set[str]] = defaultdict(set)
    for path in files:
        original_group_splits[source_id(path)].add(path.parts[-3])  # train / valid / test
    leaking_groups = sum(1 for splits in original_group_splits.values() if len(splits) > 1)

    audit = {
        "source_images": len(files),
        "invalid_removed": len(invalid),
        "phash_within_class_removed": len(within_removed),
        "cross_class_conflict_paths_removed": len(cross_paths),
        "cross_class_pairs": len(cross_pairs),
        "final_images": len(final),
        "original_split_source_groups_crossing_splits": leaking_groups,
        "invalid_examples": invalid[:20],
        "within_duplicate_examples": within_removed[:20],
        "cross_class_examples": cross_pairs[:20],
    }
    return final, audit


def _read_annotation_sheet(path: Path, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet)
    dark_column = "dark spots(0-5)" if "dark spots(0-5)" in df.columns else "pigmentation(0-5)"
    score_columns = [
        "Acne_Severity (0-5)",
        "Redness Severity (0-5)",
        "Open pores (0-5)",
        "Fine line around eyes(0-5)",
        "Wrinkes on forehead(0-5)",
        "Dehydration (0-5)(5 very dehydrated)",
        dark_column,
    ]
    # One source row contains 11 in a documented 0-5 field.  Clip all severity
    # columns to the documented range instead of letting an impossible value pass.
    for column in score_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce").clip(0, 5)

    out = pd.DataFrame({"Image_ID": df["Image_ID"].astype(str)})
    out["skin_type"] = out["Image_ID"].str.extract(r"^(oily|dry|normal|combination)", expand=False)
    out["source_group"] = out["Image_ID"].map(source_id)

    # A score of 1 is a trace/mild label and is noisy in this hand-labeled file.
    # Use >=2 as the reproducible binary threshold for a meaningful concern.
    out["acne"] = (df["Acne_Severity (0-5)"] >= 2).astype(int)
    out["dark_spots"] = (df[dark_column] >= 2).astype(int)
    out["redness"] = (df["Redness Severity (0-5)"] >= 2).astype(int)
    out["large_pores"] = (df["Open pores (0-5)"] >= 2).astype(int)
    out["wrinkles"] = (
        df[["Fine line around eyes(0-5)", "Wrinkes on forehead(0-5)"]].max(axis=1) >= 2
    ).astype(int)
    out["dryness"] = (df["Dehydration (0-5)(5 very dehydrated)"] >= 2).astype(int)
    return out


def build_labeled_frame(final_paths: list[Path]) -> tuple[pd.DataFrame, dict]:
    annotations = pd.concat(
        [
            _read_annotation_sheet(MAIN / "skinalaysis_labeling_train1.xlsx", "train"),
            _read_annotation_sheet(MAIN / "skinanalysis_valid1.xlsx", "valid"),
        ],
        ignore_index=True,
    )

    # Repeated annotations of the same source image sometimes disagree.  At the
    # source-group level, retain a positive if any repeated annotation is >=2.
    annotation_groups = annotations.groupby(["skin_type", "source_group"], as_index=False)[CONCERNS].max()
    lookup = {
        (row.skin_type, row.source_group): {c: int(getattr(row, c)) for c in CONCERNS}
        for row in annotation_groups.itertuples(index=False)
    }

    rows = []
    for path in final_paths:
        skin_type = path.parent.name
        group = source_id(path)
        labels = lookup.get((skin_type, group))
        relative = Path("data/raw/vision/skin_type_classification_dataset") / path.relative_to(MAIN)
        row = {
            "filepath": relative.as_posix(),
            "skin_type": skin_type,
            "source_group": group,
            "has_concern_labels": labels is not None,
        }
        for concern in CONCERNS:
            row[concern] = labels[concern] if labels is not None else np.nan
        rows.append(row)

    return pd.DataFrame(rows), {
        "annotation_rows": len(annotations),
        "annotation_unique_groups": len(annotation_groups),
    }


def _split_one_stratum(subset: pd.DataFrame) -> tuple[dict[str, str], dict[str, int]]:
    groups = subset.groupby("source_group").size().reset_index(name="n")
    ids = groups["source_group"].tolist()
    seed = SEED + sum(map(ord, str(subset["skin_type"].iloc[0]))) + int(subset["has_concern_labels"].iloc[0])
    random.Random(seed).shuffle(ids)

    sizes = dict(zip(groups["source_group"], groups["n"]))
    total = int(groups["n"].sum())
    targets = {"train": 0.70 * total, "val": 0.15 * total, "test": 0.15 * total}
    counts = {"train": 0, "val": 0, "test": 0}
    allocation: dict[str, str] = {}

    for i, group in enumerate(ids):
        remaining_groups = len(ids) - i
        empty = [split for split, count in counts.items() if count == 0]
        if remaining_groups <= len(empty) and empty:
            split = empty[0]
        else:
            remaining_target = {split: targets[split] - counts[split] for split in counts}
            split = max(remaining_target, key=remaining_target.get)
        allocation[group] = split
        counts[split] += int(sizes[group])
    return allocation, counts


def assign_grouped_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    allocation: dict[tuple[str, str], str] = {}
    split_audit = {}
    for (skin_type, annotated), subset in df.groupby(["skin_type", "has_concern_labels"], dropna=False):
        local, counts = _split_one_stratum(subset)
        allocation.update({(skin_type, group): split for group, split in local.items()})
        split_audit[f"{skin_type}|annotated={annotated}"] = counts
    df = df.copy()
    df["split"] = [allocation[(row.skin_type, row.source_group)] for row in df.itertuples(index=False)]
    return df, split_audit


def build_auxiliary_defects() -> pd.DataFrame:
    if not (DEFECTS / "skin_defects.csv").exists():
        return pd.DataFrame()
    raw = pd.read_csv(DEFECTS / "skin_defects.csv")
    rows = []
    for row in raw.itertuples(index=False):
        if row.type == "bags":
            continue  # outside the frozen 6-concern label space
        for view in ["front", "left_side", "right_side"]:
            relative_source = str(getattr(row, view)).lstrip("/")
            record = {
                "filepath": (Path("data/raw/vision/skin_defects/files") / relative_source).as_posix(),
                "source_group": f"defect_person_{row.id}",
                "skin_type": np.nan,
            }
            for concern in CONCERNS:
                record[concern] = np.nan
            if row.type == "acne":
                record["acne"] = 1
            elif row.type == "redness":
                record["redness"] = 1
            rows.append(record)
    return pd.DataFrame(rows)


def write_distribution(df: pd.DataFrame, audit: dict, aux: pd.DataFrame) -> None:
    lines = [
        "# Vision Data Cleaning & Class Distribution",
        "",
        f"- Source skin-type images: **{audit['source_images']:,}**",
        f"- Invalid/small/non-RGB removed: **{audit['invalid_removed']:,}**",
        f"- Within-class pHash near-duplicates removed (Hamming <= 5): **{audit['phash_within_class_removed']:,}**",
        f"- Cross-class near-identical label conflicts removed: **{audit['cross_class_conflict_paths_removed']:,} images across {audit['cross_class_pairs']:,} pairs**",
        f"- Final main-dataset images: **{len(df):,}**",
        f"- Spreadsheet annotations: **{audit['annotation_rows']:,} rows / {audit['annotation_unique_groups']:,} unique labeled source groups**",
        "- Concern binarization: documented 0-5 scores were clipped to [0,5]; scores **>=2 => 1**, scores 0-1 => 0.",
        "- Unannotated concern cells are intentionally left blank rather than falsely labeled 0.",
        "- All augmented copies sharing the same source-image ID are kept in one split.",
        "- Person identity is not provided, so true person-level grouping cannot be guaranteed.",
        "",
        "## Split sizes",
        "",
    ]
    for split in ["train", "val", "test"]:
        part = df[df["split"] == split]
        lines += [f"### {split.title()} — {len(part):,} images", "", "| skin_type | images | concern-labeled images |", "|---|---:|---:|"]
        for skin_type in SKIN_TYPES:
            subset = part[part["skin_type"] == skin_type]
            lines.append(f"| {skin_type} | {len(subset):,} | {int(subset['has_concern_labels'].sum()):,} |")
        lines += ["", "| concern | positive | negative | missing |", "|---|---:|---:|---:|"]
        for concern in CONCERNS:
            values = pd.to_numeric(part[concern], errors="coerce")
            lines.append(
                f"| {concern} | {int((values == 1).sum()):,} | {int((values == 0).sum()):,} | {int(values.isna().sum()):,} |"
            )
        lines.append("")

    if not aux.empty:
        lines += [
            "## Auxiliary Skin-Defects Dataset",
            "",
            "- The source has 90 images / 30 people: acne, eye-bags, and redness, each with 3 views.",
            "- Eye-bags were dropped because that label is outside the frozen concern space.",
            f"- **{len(aux):,}** acne/redness images remain in `vision_concern_aux.csv`.",
            "- They are not merged into the main CSVs because this source has no skin-type label.",
            "",
        ]
    (OUT / "class_distribution.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    final_paths, audit = audit_and_dedupe()
    frame, annotation_audit = build_labeled_frame(final_paths)
    frame, split_audit = assign_grouped_splits(frame)
    audit.update(annotation_audit)
    audit["split_audit"] = split_audit
    audit["split_counts"] = {k: int(v) for k, v in frame["split"].value_counts().to_dict().items()}
    audit["final_skin_type_counts"] = {k: int(v) for k, v in frame["skin_type"].value_counts().to_dict().items()}
    audit["annotated_images_after_source_group_propagation"] = int(frame["has_concern_labels"].sum())

    output_columns = ["filepath", "skin_type"] + CONCERNS
    for split in ["train", "val", "test"]:
        frame.loc[frame["split"] == split, output_columns].to_csv(OUT / f"vision_{split}.csv", index=False)

    # Keep a richer local audit CSV; the training code only consumes the 3 files above.
    frame.to_csv(OUT / "vision_audit_labels.csv", index=False)
    aux = build_auxiliary_defects()
    if not aux.empty:
        aux.to_csv(OUT / "vision_concern_aux.csv", index=False)
    audit["aux_concern_images"] = len(aux)

    write_distribution(frame, audit, aux)
    (OUT / "vision_final_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
