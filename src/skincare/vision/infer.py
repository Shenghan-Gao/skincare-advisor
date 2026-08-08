"""Wrap the trained CNN behind the SkinAnalysis contract.

自描述 checkpoint:架构参数(kind / backbone)从 checkpoint 里读,不从当前代码猜。
这样队友换 backbone、调宽度重训,你这边不用改任何代码就能加载。
"""
import io

import torch
from PIL import Image
from app.schemas import ConcernScore, SkinAnalysis, SkinType
from skincare.config import CONCERNS, SKIN_TYPES
from skincare.vision.data import build_transforms
from skincare.vision.model import build_model


class CheckpointMismatch(RuntimeError):
    """标签空间对不上 —— 这是最隐蔽的一类不兼容,单独报错说清楚。"""


class SkinClassifier:
    def __init__(self, ckpt_path: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        cfg = ck.get("config", {})
        kind = ck.get("kind", "transfer")

        # ---- 从 checkpoint 自身重建架构,而不是相信当前代码的默认值 ----
        kw = {}
        if kind == "transfer":
            kw["backbone"] = cfg.get("backbone", "resnet18")
            kw["pretrained"] = False        # 权重来自 checkpoint,不必再下载
        self.model = build_model(kind, **kw)

        # ---- 先查标签空间,再 load,把错误信息说人话 ----
        sd = ck["state_dict"]
        n_concern = sd["head_concern.weight"].shape[0]
        n_type = sd["head_type.weight"].shape[0]
        if n_concern != len(CONCERNS) or n_type != len(SKIN_TYPES):
            raise CheckpointMismatch(
                f"标签空间不一致 —— checkpoint 是 {n_type} 个肤质 / {n_concern} 个关注点,"
                f"当前 config.py 是 {len(SKIN_TYPES)} / {len(CONCERNS)}。\n"
                f"说明有人改过 config.py 的 SKIN_TYPES / CONCERNS。"
                f"这会同时影响 schemas、奖励函数和已生成的训练数据 —— 先统一标签空间再重训。"
            )

        try:
            self.model.load_state_dict(sd)
        except RuntimeError as e:
            raise CheckpointMismatch(
                f"权重与架构不匹配。checkpoint 记录 kind={kind}, "
                f"backbone={cfg.get('backbone', 'n/a')}。\n"
                f"通常说明训练用的 model.py 与当前 model.py 不是同一版 —— "
                f"让训练者确认基于哪个 git tag,或让他连同 model.py 一起交回。\n原始错误: {e}"
            ) from e

        self.model.to(self.device).eval()
        self.tf = build_transforms(train=False)
        self.version = f"{kind}:{cfg.get('backbone', '-')}:{cfg.get('run_name', '-')}"

    @torch.no_grad()
    def predict_bytes(self, raw: bytes) -> SkinAnalysis:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        x = self.tf(img).unsqueeze(0).to(self.device)
        lt, lc = self.model(x)
        pt = torch.softmax(lt, 1)[0]
        pc = torch.sigmoid(lc)[0]
        idx = int(pt.argmax())
        return SkinAnalysis(
            skin_type=SkinType(SKIN_TYPES[idx]),
            skin_type_confidence=float(pt[idx]),
            concerns=[ConcernScore(concern=c, score=float(pc[i])) for i, c in enumerate(CONCERNS)],
            model_version=self.version,
        )
