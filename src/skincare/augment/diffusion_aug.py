"""生成式图像增强(Module 7 / Assignment 3-4)—— 组员 C 主攻。

目标不是"生成好看的皮肤图",而是**用消融实验证明生成式增强能提升分类性能**。
报告里最有说服力的一句话是:
    "加入 N 张扩散合成样本后,少数类 macro-F1 从 0.61 提升到 0.68"

    python -m skincare.augment.diffusion_aug --concern acne --n 200
"""
import argparse
from pathlib import Path

from skincare.config import PROCESSED, RAW


def generate(concern: str, n: int, out_dir: Path, strength: float = 0.6):
    """img2img 增强:以真实少数类图片为起点做扩散重绘,保留皮肤纹理语义。

    TODO(C):
      1. 用 diffusers 加载 StableDiffusionImg2ImgPipeline
      2. 从该 concern 的真实样本里采样起点图
      3. prompt 用临床描述性措辞(不要艺术化措辞),strength 0.5-0.7 之间试
      4. 存到 out_dir,文件名带 synth_ 前缀便于溯源
    比从零训 GAN 稳得多,一天内能出结果。
    """
    raise NotImplementedError


def build_augmented_csv(orig_csv: str, synth_dir: Path, out_csv: str):
    """把合成样本并进训练集,生成 vision_train_aug.csv 交给组员 B 重训。

    ⚠️ 只增强 train,**绝不动 val/test** —— 否则指标失去意义。
    """
    raise NotImplementedError


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--concern", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default=str(RAW / "synthetic"))
    a = ap.parse_args()
    generate(a.concern, a.n, Path(a.out))
