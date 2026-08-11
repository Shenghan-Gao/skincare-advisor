"""Search concern thresholds on validation data and save a calibrated checkpoint."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from skincare.config import CONCERNS, PROCESSED
from skincare.vision.calibration import (
    bootstrap_concern_macro_f1,
    concern_metrics,
    search_concern_thresholds,
)
from skincare.vision.data import SkinDataset
from skincare.vision.model import build_model


def choose_device(requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def predict(model, loader, device):
    targets, probabilities = [], []
    for images, _, y_concern in loader:
        _, logits = model(images.to(device))
        targets.append(y_concern.numpy())
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(targets), np.concatenate(probabilities)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val-csv", default=str(PROCESSED / "vision_val.csv"))
    parser.add_argument("--output-checkpoint")
    parser.add_argument("--output-report")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_checkpoint = Path(args.output_checkpoint or checkpoint_path.with_name(
        f"{checkpoint_path.stem}_calibrated.pt"
    ))
    output_report = Path(args.output_report or checkpoint_path.with_name(
        f"{checkpoint_path.stem}_calibration.json"
    ))
    device = choose_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    kind = checkpoint.get("kind", "transfer")
    kwargs = {}
    if kind == "transfer":
        kwargs = {"backbone": config.get("backbone", "resnet18"), "pretrained": False}
    model = build_model(kind, **kwargs)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()

    loader = DataLoader(
        SkinDataset(args.val_csv, train=False), batch_size=args.batch_size,
        shuffle=False, num_workers=0,
    )
    targets, probabilities = predict(model, loader, device)
    default_thresholds = np.full(len(CONCERNS), 0.5)
    thresholds = search_concern_thresholds(targets, probabilities)
    report = {
        "source_checkpoint": str(checkpoint_path),
        "validation_csv": args.val_csv,
        "threshold_search": {
            "grid_min": 0.05, "grid_max": 0.95, "grid_step": 0.01,
            "tie_break": "closest_to_0.5_then_lower",
        },
        "default_0_5": concern_metrics(targets, probabilities, default_thresholds),
        "calibrated": concern_metrics(targets, probabilities, thresholds),
        "bootstrap_95_ci": bootstrap_concern_macro_f1(
            targets, probabilities, thresholds,
            samples=args.bootstrap_samples, seed=args.seed,
        ),
        "thresholds": {
            name: float(thresholds[index]) for index, name in enumerate(CONCERNS)
        },
    }
    checkpoint["concern_thresholds"] = thresholds.tolist()
    checkpoint["calibration"] = report
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_checkpoint)
    output_report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "device": device,
        "default_macro_f1": report["default_0_5"]["macro_f1"],
        "calibrated_macro_f1": report["calibrated"]["macro_f1"],
        "bootstrap_95_ci": report["bootstrap_95_ci"],
        "thresholds": report["thresholds"],
        "output_checkpoint": str(output_checkpoint),
        "output_report": str(output_report),
    }, indent=2))


if __name__ == "__main__":
    main()
