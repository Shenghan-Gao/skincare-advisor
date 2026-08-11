"""Wrap the trained CNN behind the SkinAnalysis contract.

Self-describing checkpoints: the architecture parameters (kind / backbone) are read out of
the checkpoint rather than guessed from the current code. That way a teammate can swap the
backbone or change the width and retrain, and this side loads it without a code change.
"""
import io

import torch
from PIL import Image

from app.schemas import ConcernScore, SkinAnalysis, SkinType
from skincare.config import CONCERNS, SKIN_TYPES
from skincare.vision.calibration import shift_logits_to_threshold
from skincare.vision.data import build_transforms
from skincare.vision.model import build_model


class CheckpointMismatch(RuntimeError):
    """Label spaces do not line up -- the most easily missed kind of incompatibility, so it
    gets its own exception type with an explicit message."""


class SkinClassifier:
    def __init__(self, ckpt_path: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        cfg = ck.get("config", {})
        kind = ck.get("kind", "transfer")

        # ---- Rebuild the architecture from the checkpoint, not from the current defaults ----
        kw = {}
        if kind == "transfer":
            kw["backbone"] = cfg.get("backbone", "resnet18")
            kw["pretrained"] = False        # weights come from the checkpoint, no download
        self.model = build_model(kind, **kw)

        # ---- Check the label space before loading, so the error message is readable ----
        sd = ck["state_dict"]
        n_concern = sd["head_concern.weight"].shape[0]
        n_type = sd["head_type.weight"].shape[0]
        if n_concern != len(CONCERNS) or n_type != len(SKIN_TYPES):
            raise CheckpointMismatch(
                f"Label spaces disagree -- the checkpoint has {n_type} skin types / "
                f"{n_concern} concerns, while the current config.py has "
                f"{len(SKIN_TYPES)} / {len(CONCERNS)}.\n"
                f"That means someone changed SKIN_TYPES / CONCERNS in config.py. "
                f"This affects the schemas, the reward functions and the training data "
                f"already generated -- agree on one label space, then retrain."
            )

        try:
            self.model.load_state_dict(sd)
        except RuntimeError as e:
            raise CheckpointMismatch(
                f"Weights do not match the architecture. The checkpoint records kind={kind}, "
                f"backbone={cfg.get('backbone', 'n/a')}.\n"
                f"This usually means the model.py used for training is not the same version "
                f"as the current one -- ask whoever trained it which git tag it was built "
                f"from, or have them hand back their model.py with it.\nOriginal error: {e}"
            ) from e

        self.model.to(self.device).eval()
        self.concern_thresholds = ck.get("concern_thresholds", [0.5] * len(CONCERNS))
        if len(self.concern_thresholds) != len(CONCERNS):
            raise CheckpointMismatch(
                f"Expected {len(CONCERNS)} concern thresholds, got "
                f"{len(self.concern_thresholds)}"
            )
        self.tf = build_transforms(train=False)
        self.version = f"{kind}:{cfg.get('backbone', '-')}:{cfg.get('run_name', '-')}"

    @torch.no_grad()
    def predict_bytes(self, raw: bytes) -> SkinAnalysis:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        x = self.tf(img).unsqueeze(0).to(self.device)
        lt, lc = self.model(x)
        pt = torch.softmax(lt, 1)[0]
        # A calibrated checkpoint stores raw per-class validation thresholds. Shift
        # logits so the unchanged SkinAnalysis.top_concerns(0.5) API applies them.
        pc = torch.sigmoid(shift_logits_to_threshold(lc, self.concern_thresholds))[0]
        idx = int(pt.argmax())
        return SkinAnalysis(
            skin_type=SkinType(SKIN_TYPES[idx]),
            skin_type_confidence=float(pt[idx]),
            concerns=[ConcernScore(concern=c, score=float(pc[i])) for i, c in enumerate(CONCERNS)],
            model_version=self.version,
        )
