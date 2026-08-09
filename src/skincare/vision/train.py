"""Train Pillar 1. Run on Colab/Kaggle GPU.

Two ways to run:
    python -m skincare.vision.train --config configs/vision_transfer.yaml   # handoff runs
    python -m skincare.vision.train --kind transfer --epochs 8              # quick manual sweep

When this is handed to a teammate they change configs/*.yaml only, never this file.
The checkpoint they return must pass scripts/verify_handoff.py.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from skincare.config import CONCERNS, MODELS, PROCESSED
from skincare.vision.data import make_loaders
from skincare.vision.model import build_model, multitask_loss


def evaluate(model, loader, device):
    """Evaluate. Concern F1 is computed only over annotated positions, mirroring the
    masked loss. Without the mask, sklearn would treat -1 as a genuine class label
    and the reported scores would be meaningless."""
    model.eval()
    tp, tt, probs, targs = [], [], [], []
    with torch.no_grad():
        for x, yt, yc in loader:
            lt, lc = model(x.to(device))
            tp += lt.argmax(1).cpu().tolist()
            tt += yt.tolist()
            probs.append(torch.sigmoid(lc).cpu().numpy())
            targs.append(yc.numpy())

    probs = np.concatenate(probs) if probs else np.zeros((0, len(CONCERNS)))
    targs = np.concatenate(targs) if targs else np.zeros((0, len(CONCERNS)))
    preds = (probs > 0.5).astype(int)

    # A concern is scored only where ground truth exists. Unlabelled entries arrive as
    # NaN (from the CSV) or -1 (from SkinDataset); neither counts as a negative example.
    per_concern, coverage = {}, {}
    for j, name in enumerate(CONCERNS):
        col = targs[:, j]
        m = np.isfinite(col) & (col >= 0)
        coverage[name] = int(m.sum())
        per_concern[name] = (f1_score(col[m].astype(int), preds[m, j],
                                      zero_division=0) if m.sum() else None)
    scored = [v for v in per_concern.values() if v is not None]

    return {
        "type_acc": sum(int(a == b) for a, b in zip(tp, tt)) / max(len(tt), 1),
        "type_macro_f1": f1_score(tt, tp, average="macro", zero_division=0),
        "concern_macro_f1": sum(scored) / len(scored) if scored else 0.0,
        "concern_f1_per_class": {k: (round(v, 4) if v is not None else None)
                                 for k, v in per_concern.items()},
        "concern_labeled_counts": coverage,
    }, (tt, tp)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="YAML config used for handoff runs. CLI flags override it.")
    ap.add_argument("--kind", choices=["transfer", "simple"])
    ap.add_argument("--backbone")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--freeze-backbone", action="store_true")
    ap.add_argument("--train-csv")
    ap.add_argument("--val-csv")
    ap.add_argument("--out")
    ap.add_argument("--run-name")
    args = ap.parse_args()

    cfg = {
        "kind": "transfer", "backbone": "resnet18", "epochs": 8, "lr": 3e-4,
        "batch_size": 32, "freeze_backbone": False,
        "train_csv": str(PROCESSED / "vision_train.csv"),
        "val_csv": str(PROCESSED / "vision_val.csv"),
        "out": str(MODELS / "vision"), "run_name": "run",
    }
    if args.config:
        import yaml
        cfg.update(yaml.safe_load(open(args.config)))
    for k, v in vars(args).items():
        if k != "config" and v not in (None, False):
            cfg[k.replace("-", "_")] = v
    return cfg


def main():
    cfg = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}\nconfig={json.dumps(cfg, indent=2)}")

    train_dl, val_dl = make_loaders(cfg["train_csv"], cfg["val_csv"], cfg["batch_size"])
    kw = {}
    if cfg["kind"] == "transfer":
        kw = {"backbone": cfg["backbone"], "freeze_backbone": cfg["freeze_backbone"]}
    model = build_model(cfg["kind"], **kw).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"])

    out_dir = Path(cfg["out"]); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{cfg['run_name']}.pt"
    history, best = [], -1.0

    for ep in range(cfg["epochs"]):
        model.train()
        total = 0.0
        for x, yt, yc in train_dl:
            opt.zero_grad()
            lt, lc = model(x.to(device))
            loss, _, _ = multitask_loss(lt, lc, yt.to(device), yc.to(device))
            loss.backward(); opt.step()
            total += loss.item()
        sched.step()
        m, _ = evaluate(model, val_dl, device)
        m["train_loss"] = total / max(len(train_dl), 1)
        history.append(m)
        print(f"epoch {ep+1}/{cfg['epochs']} {m}")

        score = m["type_macro_f1"] + m["concern_macro_f1"]
        if score > best:
            best = score
            torch.save({"state_dict": model.state_dict(), "kind": cfg["kind"],
                        "metrics": m, "config": cfg}, ckpt_path)
            print(f"  saved -> {ckpt_path}")

    # Handoff artefact: the teammate returns this JSON alongside the .pt checkpoint.
    _, (y_true, y_pred) = evaluate(model, val_dl, device)
    report = {
        "run_name": cfg["run_name"], "config": cfg, "history": history,
        "best_score": best,
        "per_class": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }
    with open(out_dir / f"{cfg['run_name']}_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\ndone. best={best:.4f}\ncheckpoint: {ckpt_path}")
    print(f"report:     {out_dir / (cfg['run_name'] + '_report.json')}")
    print(f"\nNext: python scripts/verify_handoff.py vision {ckpt_path}")


if __name__ == "__main__":
    main()
