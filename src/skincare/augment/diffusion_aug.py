"""Generative image augmentation (Module 7 / Assignment 3-4) -- led by member C.

The goal is not "generate good-looking skin images" but **to prove with an ablation that
generative augmentation improves classification performance**. The most convincing
sentence in the report looks like:
    "after adding N diffusion-synthesised samples, minority-class macro-F1 rose from
     0.61 to 0.68"

    python -m skincare.augment.diffusion_aug --concern acne --n 200
"""
import argparse
from pathlib import Path

from skincare.config import PROCESSED, RAW


def generate(concern: str, n: int, out_dir: Path, strength: float = 0.6):
    """img2img augmentation: start from a real minority-class photo and let diffusion
    repaint it, which keeps the semantics of the skin texture intact.

    TODO(C):
      1. load StableDiffusionImg2ImgPipeline via diffusers
      2. sample the starting images from that concern's real samples
      3. write the prompt in clinical, descriptive terms (not artistic ones); try a
         strength between 0.5 and 0.7
      4. save into out_dir with a synth_ filename prefix so samples stay traceable
    Far more reliable than training a GAN from scratch, and it produces results within a day.
    """
    raise NotImplementedError


def build_augmented_csv(orig_csv: str, synth_dir: Path, out_csv: str):
    """Merge the synthetic samples into the training set and write vision_train_aug.csv
    for member B to retrain on.

    ⚠️ Augment train only, **never touch val/test** -- otherwise the metrics mean nothing.
    """
    raise NotImplementedError


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--concern", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default=str(RAW / "synthetic"))
    a = ap.parse_args()
    generate(a.concern, a.n, Path(a.out))
