"""Generative image augmentation and fallback oversampling for member C.

This module reproduces the augmentation workflow used in the project:
1. Generate Stable Diffusion v1.5 img2img pilots from *training* images only.
2. Manually QC generated images before any synthetic sample can enter training.
3. If diffusion QC fails, use minority-class oversampling as the fallback ablation.

Examples
--------
Generate a diffusion pilot (does NOT modify the training CSV):
    python -m skincare.augment.diffusion_aug --mode diffusion \
        --target acne --n 20 --strength 0.5

After manual QC, set ``accepted`` to 1/0 in the generated ``metadata.csv`` and build
an augmented training CSV containing only accepted synthetic samples:
    python -m skincare.augment.diffusion_aug --mode build-synthetic-csv \
        --target acne

Fallback used for the reported ablation:
    python -m skincare.augment.diffusion_aug --mode oversample \
        --target combination --n 200

Validation/test CSVs are never read or modified by this module.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from skincare.config import CONCERNS, PROCESSED, RAW, ROOT, SEED, SKIN_TYPES

DEFAULT_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DEFAULT_PROMPTS = {
    "acne": (
        "clinical facial skin photograph with mild acne, realistic skin texture, "
        "neutral lighting"
    ),
    "dark_spots": (
        "clinical facial skin photograph with visible dark spots and post-acne marks, "
        "realistic skin texture, neutral lighting"
    ),
    "redness": (
        "clinical facial skin photograph with mild facial redness, realistic skin texture, "
        "neutral lighting"
    ),
    "large_pores": (
        "clinical facial skin photograph with visible enlarged pores, realistic skin texture, "
        "neutral lighting"
    ),
    "wrinkles": (
        "clinical facial skin photograph with visible fine lines and wrinkles, realistic skin "
        "texture, neutral lighting"
    ),
    "dryness": (
        "clinical facial skin photograph with visible dry skin texture, realistic skin texture, "
        "neutral lighting"
    ),
    "oily": (
        "clinical facial skin photograph of oily skin, realistic skin texture, neutral lighting"
    ),
    "dry": "clinical facial skin photograph of dry skin, realistic skin texture, neutral lighting",
    "combination": (
        "clinical facial skin photograph of combination skin, realistic skin texture, "
        "neutral lighting"
    ),
    "normal": (
        "clinical facial skin photograph of normal skin, realistic skin texture, neutral lighting"
    ),
}
NEGATIVE_PROMPT = (
    "cartoon, illustration, painting, text, watermark, distorted face, extra face, "
    "extra eyes, face paint, facial markings, dots, water droplets, cream, foam, mask, "
    "stickers, artificial skin texture, plastic skin"
)


def _training_pool(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Return labelled training rows eligible to seed augmentation for ``target``."""
    if target in SKIN_TYPES:
        pool = df[df["skin_type"] == target].copy()
    elif target in CONCERNS:
        labels = pd.to_numeric(df[target], errors="coerce")
        pool = df[labels == 1].copy()
    else:
        allowed = ", ".join(SKIN_TYPES + CONCERNS)
        raise ValueError(f"Unknown target {target!r}. Expected one of: {allowed}")

    pool["_abs_path"] = pool["filepath"].map(lambda p: ROOT / str(p))
    pool = pool[pool["_abs_path"].map(lambda p: p.exists())].copy()
    if pool.empty:
        raise ValueError(f"No usable training images found for target={target!r}")
    return pool


def _repo_relative(path: Path) -> str:
    """Return a repository-relative path required by the vision CSV contract."""
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT.resolve()))
    except ValueError as exc:
        raise ValueError(
            f"Synthetic output must live inside the repository root {ROOT}; got {path}"
        ) from exc


def generate(
    concern: str,
    n: int,
    out_dir: Path,
    strength: float = 0.6,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    seed: int = SEED,
    prompt: str | None = None,
) -> pd.DataFrame:
    """Generate img2img samples from real *training* images.

    ``concern`` is kept as the argument name for backwards compatibility with the
    original project command, but it may be either a concern label (for example
    ``acne``) or a skin type (for example ``combination``).

    This function only generates pilot images and metadata. It intentionally does
    **not** append synthetic samples to the training CSV. Each generated row starts
    with an empty ``accepted`` field so a human QC decision is required before a
    synthetic sample can be admitted by :func:`build_augmented_csv`.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= strength <= 1:
        raise ValueError("strength must be between 0 and 1")

    try:
        import torch
        from diffusers import StableDiffusionImg2ImgPipeline
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Diffusion augmentation needs torch, Pillow and diffusers. "
            "In Colab install: pip install diffusers transformers accelerate safetensors"
        ) from exc

    train_csv = PROCESSED / "vision_train.csv"
    df = pd.read_csv(train_csv)
    pool = _training_pool(df, concern)
    sampled = pool.sample(n=n, replace=len(pool) < n, random_state=seed).reset_index(drop=True)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt = prompt or DEFAULT_PROMPTS.get(
        concern,
        (
            "clinical facial skin photograph with visible "
            f"{concern.replace('_', ' ')}, realistic skin texture"
        ),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to(device)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    records: list[dict] = []
    for i, row in sampled.iterrows():
        source_rel = str(row["filepath"])
        source_abs = ROOT / source_rel
        image = Image.open(source_abs).convert("RGB").resize((512, 512))

        generator = torch.Generator(device=device).manual_seed(seed + i)
        result = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            image=image,
            strength=strength,
            guidance_scale=7.5,
            num_inference_steps=30,
            generator=generator,
        ).images[0]

        target_dir = out_dir / str(row["skin_type"])
        target_dir.mkdir(parents=True, exist_ok=True)
        out_name = f"synth_{concern}_{i:04d}_{Path(source_rel).stem}.jpg"
        out_abs = target_dir / out_name
        result.save(out_abs, quality=95)

        records.append(
            {
                "filepath": _repo_relative(out_abs),
                "source_filepath": source_rel,
                "target": concern,
                "strength": strength,
                "model_id": model_id,
                "prompt": prompt,
                "seed": seed + i,
                "accepted": pd.NA,
                "qc_notes": "",
            }
        )

    metadata = pd.DataFrame(records)
    metadata.to_csv(out_dir / "metadata.csv", index=False)
    with (out_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "target": concern,
                "n": n,
                "strength": strength,
                "model_id": model_id,
                "seed": seed,
                "prompt": prompt,
                "negative_prompt": NEGATIVE_PROMPT,
                "source_csv": str(train_csv.relative_to(ROOT)),
                "qc_required_before_training": True,
            },
            f,
            indent=2,
        )
    return metadata


def _accepted_mask(series: pd.Series) -> pd.Series:
    """Normalize common manual-QC values to a boolean accepted mask."""
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.isin({"1", "true", "yes", "y", "accept", "accepted"})


def build_augmented_csv(orig_csv: str, synth_dir: Path, out_csv: str) -> pd.DataFrame:
    """Append *human-accepted* synthetic samples to a training CSV.

    ``metadata.csv`` must include an ``accepted`` column filled during manual QC.
    Only accepted rows are appended. Labels are copied from each synthetic image's
    ``source_filepath``, preserving unknown concern labels (NaN) exactly.
    """
    orig = pd.read_csv(orig_csv)
    metadata_path = Path(synth_dir) / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
    metadata = pd.read_csv(metadata_path)

    required = {"filepath", "source_filepath", "accepted"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"metadata.csv is missing columns: {sorted(missing)}")

    accepted = metadata[_accepted_mask(metadata["accepted"])].copy()
    if accepted.empty:
        merged = orig.copy()
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_csv, index=False)
        return merged

    source_rows = orig.set_index("filepath", drop=False)
    synth_rows = []
    for _, item in accepted.iterrows():
        source = str(item["source_filepath"])
        if source not in source_rows.index:
            raise ValueError(f"Synthetic source is not in the training CSV: {source}")
        row = source_rows.loc[source].copy()
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0].copy()
        row["filepath"] = str(item["filepath"])
        synth_rows.append(row)

    synth_df = pd.DataFrame(synth_rows, columns=orig.columns)
    merged = pd.concat([orig, synth_df], ignore_index=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    return merged


def build_oversampled_csv(
    orig_csv: str,
    target: str,
    n: int,
    out_csv: str,
    *,
    seed: int = SEED,
) -> pd.DataFrame:
    """Create the fallback minority-class oversampling training CSV.

    This reproduces the reported negative ablation: ``combination`` +200 rows
    (2,718 -> 2,918). No image bytes are created; sampled rows continue to point to
    original training images, so downstream stochastic transforms can provide image
    variation during training. Unknown concern labels remain NaN.
    """
    if n <= 0:
        raise ValueError("n must be positive")

    orig = pd.read_csv(orig_csv)
    if target in SKIN_TYPES:
        pool = orig[orig["skin_type"] == target].copy()
    elif target in CONCERNS:
        labels = pd.to_numeric(orig[target], errors="coerce")
        pool = orig[labels == 1].copy()
    else:
        allowed = ", ".join(SKIN_TYPES + CONCERNS)
        raise ValueError(f"Unknown target {target!r}. Expected one of: {allowed}")
    if pool.empty:
        raise ValueError(f"No training rows found for target={target!r}")

    extra = pool.sample(n=n, replace=len(pool) < n, random_state=seed)
    extra = extra[orig.columns]
    merged = pd.concat([orig, extra], ignore_index=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode",
        choices=["diffusion", "build-synthetic-csv", "oversample"],
        default="diffusion",
    )
    ap.add_argument("--target", "--concern", dest="target", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--strength", type=float, default=0.6)
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=str(RAW / "synthetic"))
    ap.add_argument("--out-csv", default=str(PROCESSED / "vision_train_aug.csv"))
    args = ap.parse_args()

    train_csv = str(PROCESSED / "vision_train.csv")
    target_dir = Path(args.out) / args.target

    if args.mode == "oversample":
        merged = build_oversampled_csv(
            train_csv,
            args.target,
            args.n,
            args.out_csv,
            seed=args.seed,
        )
        print(f"saved oversampled training CSV: {args.out_csv} ({len(merged)} rows)")
        return

    if args.mode == "build-synthetic-csv":
        merged = build_augmented_csv(train_csv, target_dir, args.out_csv)
        print(f"saved QC-filtered augmented training CSV: {args.out_csv} ({len(merged)} rows)")
        return

    metadata = generate(
        args.target,
        args.n,
        target_dir,
        strength=args.strength,
        model_id=args.model_id,
        seed=args.seed,
    )
    print(f"saved {len(metadata)} diffusion pilot samples to: {target_dir}")
    print(f"manual QC required: edit {target_dir / 'metadata.csv'} before training")


if __name__ == "__main__":
    main()
