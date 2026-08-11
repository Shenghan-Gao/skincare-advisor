"""Evaluate trained vision checkpoints and create handoff-ready plots.

Example:
    python scripts/evaluate_vision_checkpoints.py \
        --checkpoint models/vision/simple_cnn_baseline.pt \
        --checkpoint models/vision/transfer_resnet18_lr3e4.pt \
        --checkpoint models/vision/transfer_resnet50_lr1e3_screen5.pt

The validation CSV is never modified. Concern metrics include only positions with
ground-truth labels, matching the masked training loss and train-time evaluation.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

from skincare.config import CONCERNS, MODELS, PROCESSED, SKIN_TYPES
from skincare.vision.data import SkinDataset
from skincare.vision.model import build_model


def load_checkpoint(path: Path, device: str):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    kind = checkpoint.get("kind", "transfer")
    kwargs = {}
    if kind == "transfer":
        kwargs = {"backbone": config.get("backbone", "resnet18"), "pretrained": False}
    model = build_model(kind, **kwargs)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint


@torch.no_grad()
def predict(model, loader, device: str):
    type_true, type_pred = [], []
    concern_true, concern_prob = [], []
    for images, y_type, y_concern in loader:
        type_logits, concern_logits = model(images.to(device))
        type_true.append(y_type.numpy())
        type_pred.append(type_logits.argmax(1).cpu().numpy())
        concern_true.append(y_concern.numpy())
        concern_prob.append(torch.sigmoid(concern_logits).cpu().numpy())
    return (
        np.concatenate(type_true),
        np.concatenate(type_pred),
        np.concatenate(concern_true),
        np.concatenate(concern_prob),
    )


def class_metrics(y_true, y_pred, labels, names):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        name: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, name in enumerate(names)
    }


def concern_metrics(targets, probabilities):
    predictions = (probabilities > 0.5).astype(int)
    result = {}
    for index, name in enumerate(CONCERNS):
        column = targets[:, index]
        valid = np.isfinite(column) & (column >= 0)
        if not valid.any():
            result[name] = {"precision": None, "recall": None, "f1": None, "support": 0}
            continue
        precision, recall, f1, _ = precision_recall_fscore_support(
            column[valid].astype(int), predictions[valid, index],
            average="binary", zero_division=0,
        )
        result[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(valid.sum()),
        }
    return result


def plot_confusion(matrix, run_name: str, output: Path):
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix, row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Recall within true class")
    ax.set(
        title=f"Skin-type confusion matrix — {run_name}",
        xlabel="Predicted class",
        ylabel="True class",
        xticks=range(len(SKIN_TYPES)),
        yticks=range(len(SKIN_TYPES)),
        xticklabels=SKIN_TYPES,
        yticklabels=SKIN_TYPES,
    )
    threshold = 0.5
    for row in range(len(SKIN_TYPES)):
        for col in range(len(SKIN_TYPES)):
            ax.text(
                col, row, f"{matrix[row, col]}\n{normalized[row, col]:.1%}",
                ha="center", va="center",
                color="white" if normalized[row, col] > threshold else "black",
                fontsize=10,
            )
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_per_class(type_result, concern_result, run_name: str, output: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    groups = [
        (axes[0], SKIN_TYPES, type_result, "Skin type"),
        (axes[1], CONCERNS, concern_result, "Concern (labelled positions only)"),
    ]
    for ax, names, metrics, title in groups:
        values = [metrics[name]["f1"] or 0.0 for name in names]
        bars = ax.bar(range(len(names)), values, color="#4C78A8")
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.set_ylabel("F1 score")
        ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02,
                    f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle(f"Per-class F1 — {run_name}")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def concern_correlation(probabilities):
    """Summarise whether concern outputs move together across the validation set."""
    matrix = np.corrcoef(probabilities, rowvar=False)
    off_diagonal = matrix[np.triu_indices(len(CONCERNS), k=1)]
    finite = off_diagonal[np.isfinite(off_diagonal)]
    return {
        "matrix": {
            name: {
                other: (float(matrix[row, col]) if np.isfinite(matrix[row, col]) else None)
                for col, other in enumerate(CONCERNS)
            }
            for row, name in enumerate(CONCERNS)
        },
        "off_diagonal": {
            "mean": float(finite.mean()) if finite.size else None,
            "min": float(finite.min()) if finite.size else None,
            "max": float(finite.max()) if finite.size else None,
            "pairs_at_least_0_8": int((finite >= 0.8).sum()),
            "finite_pairs": int(finite.size),
            "total_pairs": len(off_diagonal),
        },
    }


def baseline_rows(type_true, concern_true):
    """Compute validation baselines from labels instead of hard-coding dataset counts."""
    majority_class = int(np.bincount(type_true, minlength=len(SKIN_TYPES)).argmax())
    majority_prediction = np.full_like(type_true, majority_class)
    type_result = class_metrics(
        type_true, majority_prediction, list(range(len(SKIN_TYPES))), SKIN_TYPES
    )
    concern_result = concern_metrics(concern_true, np.ones_like(concern_true))
    concern_scores = [
        value["f1"] for value in concern_result.values() if value["f1"] is not None
    ]
    blank = {
        "kind": "baseline", "backbone": None, "learning_rate": None,
        "checkpoint_mb": None, "train_type_accuracy": None,
        "generalization_gap_accuracy": None, "mean_macro_f1": None,
        "concern_correlation_mean": None, "concern_correlation_max": None,
        "concern_correlation_pairs_at_least_0_8": None,
    }
    return [
        {
            "run_name": f"majority_type_baseline_{SKIN_TYPES[majority_class]}",
            "row_type": "baseline", **blank,
            "type_accuracy": float((type_true == majority_prediction).mean()),
            "type_macro_f1": float(np.mean([value["f1"] for value in type_result.values()])),
            "concern_macro_f1": None,
        },
        {
            "run_name": "all_positive_concern_baseline",
            "row_type": "baseline", **blank,
            "type_accuracy": None, "type_macro_f1": None,
            "concern_macro_f1": float(np.mean(concern_scores)),
        },
    ]


def plot_comparison(rows, baselines, output: Path):
    labels = [row["run_name"] for row in rows]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 5.5))
    series = [
        ("Skin-type accuracy", [row["type_accuracy"] for row in rows], "#4C78A8"),
        ("Skin-type macro-F1", [row["type_macro_f1"] for row in rows], "#F58518"),
        ("Concern macro-F1", [row["concern_macro_f1"] for row in rows], "#54A24B"),
    ]
    for offset, (name, values, color) in enumerate(series):
        bars = ax.bar(x + (offset - 1) * width, values, width, label=name, color=color)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012,
                    f"{value:.3f}", ha="center", va="bottom", fontsize=8, rotation=90)
    type_baseline, concern_baseline = baselines
    ax.axhline(
        type_baseline["type_accuracy"], color="#4C78A8", linestyle="--", alpha=0.75,
        label=f"Majority type baseline ({type_baseline['type_accuracy']:.3f})",
    )
    ax.axhline(
        concern_baseline["concern_macro_f1"], color="#54A24B", linestyle=":", alpha=0.9,
        label=f"All-positive concern baseline ({concern_baseline['concern_macro_f1']:.3f})",
    )
    ax.set_ylim(0, 0.82)
    ax.set_ylabel("Score")
    ax.set_title("Vision checkpoint comparison on validation set")
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", ncol=3)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--train-csv", default=str(PROCESSED / "vision_train.csv"))
    parser.add_argument("--val-csv", default=str(PROCESSED / "vision_val.csv"))
    parser.add_argument("--output", default=str(MODELS / "vision" / "evaluation"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    val_loader = DataLoader(
        SkinDataset(args.val_csv, train=False), batch_size=args.batch_size,
        shuffle=False, num_workers=0,
    )
    train_loader = DataLoader(
        SkinDataset(args.train_csv, train=False), batch_size=args.batch_size,
        shuffle=False, num_workers=0,
    )
    rows, full_report, baselines = [], {}, None
    for raw_path in args.checkpoint:
        path = Path(raw_path)
        run_name = path.stem
        print(f"evaluating {run_name} on {device}", flush=True)
        model, checkpoint = load_checkpoint(path, device)
        type_true, type_pred, concern_true, concern_prob = predict(model, val_loader, device)
        train_type_true, train_type_pred, _, _ = predict(model, train_loader, device)

        type_result = class_metrics(
            type_true, type_pred, list(range(len(SKIN_TYPES))), SKIN_TYPES
        )
        concern_result = concern_metrics(concern_true, concern_prob)
        type_accuracy = float((type_true == type_pred).mean())
        type_macro_f1 = float(np.mean([value["f1"] for value in type_result.values()]))
        valid_concern_f1 = [
            value["f1"] for value in concern_result.values() if value["f1"] is not None
        ]
        concern_macro_f1 = float(np.mean(valid_concern_f1))
        train_type_accuracy = float((train_type_true == train_type_pred).mean())
        correlation = concern_correlation(concern_prob)
        matrix = confusion_matrix(type_true, type_pred, labels=range(len(SKIN_TYPES)))

        if baselines is None:
            baselines = baseline_rows(type_true, concern_true)

        row = {
            "run_name": run_name,
            "row_type": "model",
            "kind": checkpoint.get("kind", "transfer"),
            "backbone": checkpoint.get("config", {}).get("backbone"),
            "learning_rate": checkpoint.get("config", {}).get("lr"),
            "checkpoint_mb": path.stat().st_size / 1024 / 1024,
            "type_accuracy": type_accuracy,
            "type_macro_f1": type_macro_f1,
            "concern_macro_f1": concern_macro_f1,
            "train_type_accuracy": train_type_accuracy,
            "generalization_gap_accuracy": train_type_accuracy - type_accuracy,
            "mean_macro_f1": (type_macro_f1 + concern_macro_f1) / 2,
            "concern_correlation_mean": correlation["off_diagonal"]["mean"],
            "concern_correlation_max": correlation["off_diagonal"]["max"],
            "concern_correlation_pairs_at_least_0_8": (
                correlation["off_diagonal"]["pairs_at_least_0_8"]
            ),
        }
        rows.append(row)
        full_report[run_name] = {
            **row,
            "type_per_class": type_result,
            "concern_per_class": concern_result,
            "concern_probability_correlation": correlation,
            "type_confusion_matrix": matrix.tolist(),
        }
        plot_confusion(matrix, run_name, output / f"{run_name}_confusion_matrix.png")
        plot_per_class(
            type_result, concern_result, run_name,
            output / f"{run_name}_per_class_f1.png",
        )

    full_report["_baselines"] = {row["run_name"]: row for row in baselines}
    with (output / "vision_evaluation.json").open("w") as handle:
        json.dump(full_report, handle, indent=2)
    with (output / "vision_model_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(baselines + rows)
    plot_comparison(rows, baselines, output / "vision_model_comparison.png")
    print(f"saved evaluation artifacts to {output}")


if __name__ == "__main__":
    main()
