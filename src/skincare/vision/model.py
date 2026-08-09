"""Pillar 1 -- Module 3 (FCNN/CNN) + Assignment 2.

Two models on purpose:
  * SimpleCNN      -> built from scratch, shows we understand conv/pool/BN/dropout
  * build_transfer -> pretrained ResNet/EfficientNet, fine-tuned (this is L2)
The report compares them. Do not delete SimpleCNN -- it is the "from first
principles" evidence the rubric rewards.
"""
import torch
import torch.nn as nn
from skincare.config import CONCERNS, SKIN_TYPES


class SimpleCNN(nn.Module):
    """Hand-rolled CNN: 4 conv blocks -> GAP -> two heads."""

    def __init__(self, n_types: int = len(SKIN_TYPES), n_concerns: int = len(CONCERNS)):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.3)
        self.head_type = nn.Linear(256, n_types)        # single-label  -> CrossEntropy
        self.head_concern = nn.Linear(256, n_concerns)  # multi-label   -> BCEWithLogits

    def forward(self, x):
        z = self.drop(self.pool(self.features(x)).flatten(1))
        return self.head_type(z), self.head_concern(z)


class TransferNet(nn.Module):
    """Pretrained backbone + the same two heads (fine-tuning = L2)."""

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True,
                 freeze_backbone: bool = False):
        super().__init__()
        import torchvision.models as tvm
        net = getattr(tvm, backbone)(weights="DEFAULT" if pretrained else None)
        if hasattr(net, "fc"):
            feat_dim, net.fc = net.fc.in_features, nn.Identity()
        else:  # efficientnet / convnext
            feat_dim, net.classifier = net.classifier[-1].in_features, nn.Identity()
        self.backbone = net
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.drop = nn.Dropout(0.3)
        self.head_type = nn.Linear(feat_dim, len(SKIN_TYPES))
        self.head_concern = nn.Linear(feat_dim, len(CONCERNS))

    def forward(self, x):
        z = self.drop(self.backbone(x))
        return self.head_type(z), self.head_concern(z)


def build_model(kind: str = "transfer", **kw) -> nn.Module:
    return SimpleCNN(**kw) if kind == "simple" else TransferNet(**kw)


def multitask_loss(type_logits, concern_logits, y_type, y_concern, w: float = 1.0):
    """Multi-task loss. Skin type uses every sample; concerns use only annotated labels.

    Why the mask is required: one source dataset may annotate acne but never annotate
    wrinkles. If those blanks were folded in as zeros, the model would learn "confirmed
    absent" from what is really "never labelled", and every sparsely annotated concern
    would be biased toward the negative class. In this dataset only ~5% of images carry
    concern annotations, so an unmasked loss would be dominated by fabricated negatives.

    An unlabelled entry may arrive either as NaN (straight from the CSV) or as the -1
    sentinel produced by ``SkinDataset``. Both are handled here by selecting the valid
    positions *before* computing BCE. Multiplying a mask onto an already-computed BCE
    tensor does not work: NaN * 0 is NaN, so the loss would still be poisoned.
    """
    ce = nn.functional.cross_entropy(type_logits, y_type)

    valid = torch.isfinite(y_concern) & (y_concern >= 0)
    if valid.any():
        bce = nn.functional.binary_cross_entropy_with_logits(
            concern_logits[valid], y_concern[valid])
    else:
        # No concern labels in this batch. Return a zero that is still connected to the
        # graph so the concern head keeps receiving (zero) gradient and DDP stays happy.
        bce = concern_logits.sum() * 0.0

    return ce + w * bce, ce.item(), bce.detach().item()
