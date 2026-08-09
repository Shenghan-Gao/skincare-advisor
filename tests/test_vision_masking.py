"""Masking of unannotated concern labels (requested by Member B's bug report).

Only ~5% of the images in this dataset carry concern annotations. An unannotated
entry reaches the loss either as NaN (straight from the CSV) or as the -1 sentinel
produced by SkinDataset. Neither may be treated as a negative example, and neither
may poison the loss or the metrics.
"""
import numpy as np
import pytest

# torch lives in the optional `vision` extra. A grader who installs only [dev,ui]
# must still get a green `make test`, so skip this module instead of failing.
# To run these locally: uv pip install -e ".[vision]"
torch = pytest.importorskip("torch", reason="install the 'vision' extra to run vision tests")

from skincare.config import CONCERNS  # noqa: E402
from skincare.vision.model import build_model, multitask_loss  # noqa: E402

NAN = float("nan")


def _batch(n=4):
    torch.manual_seed(0)
    return (torch.randn(n, 4, requires_grad=True),
            torch.randn(n, len(CONCERNS), requires_grad=True),
            torch.arange(n) % 4)


@pytest.mark.parametrize("blank", [NAN, -1.0])
def test_partial_labels_give_finite_loss(blank):
    """1. Partially annotated targets must produce a finite loss."""
    lt, lc, yt = _batch()
    y = torch.tensor([[1.0, 0.0, 1.0, blank, blank, blank]] * 4)
    loss, ce, bce = multitask_loss(lt, lc, yt, y)
    assert torch.isfinite(loss), f"loss is not finite for blank={blank}"
    assert np.isfinite(bce) and np.isfinite(ce)


@pytest.mark.parametrize("blank", [NAN, -1.0])
def test_masked_bce_equals_manual_selection(blank):
    """2. Masked BCE must equal BCE computed on the hand-selected valid positions."""
    lt, lc, yt = _batch()
    y = torch.tensor([[1.0, 0.0, blank, 1.0, blank, 0.0]] * 4)
    _, _, bce = multitask_loss(lt, lc, yt, y)

    valid = torch.isfinite(y) & (y >= 0)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        lc[valid], y[valid]).item()
    assert bce == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("blank", [NAN, -1.0])
def test_all_unlabeled_batch_still_backprops(blank):
    """3. A batch with no concern labels must still be differentiable."""
    lt, lc, yt = _batch()
    y = torch.full((4, len(CONCERNS)), blank)
    loss, _, bce = multitask_loss(lt, lc, yt, y)
    assert torch.isfinite(loss) and bce == 0.0
    loss.backward()
    assert lc.grad is not None and torch.isfinite(lc.grad).all()


def test_skin_type_still_trains_when_concerns_missing():
    """4. Skin type uses every sample even when no concern label exists."""
    lt, lc, yt = _batch()
    y = torch.full((4, len(CONCERNS)), NAN)
    loss, ce, _ = multitask_loss(lt, lc, yt, y)
    loss.backward()
    assert lt.grad is not None and lt.grad.abs().sum() > 0, "skin type head got no gradient"
    assert ce > 0


def test_unlabeled_positions_receive_no_gradient():
    """6. An unknown label must not act as a negative example: zero gradient there."""
    lt, lc, yt = _batch()
    y = torch.tensor([[1.0, 0.0, 1.0, NAN, NAN, NAN]] * 4)
    loss, _, _ = multitask_loss(lt, lc, yt, y)
    loss.backward()
    assert lc.grad[:, :3].abs().sum() > 0, "annotated concerns should get gradient"
    assert lc.grad[:, 3:].abs().sum() == 0, "unannotated concerns must get zero gradient"


def test_evaluation_ignores_nan_and_reports_coverage():
    """5 & 7. Evaluation skips unlabeled entries instead of raising on NaN,
    and reports which concerns actually had ground truth."""
    from skincare.vision.train import evaluate

    class _Loader:
        def __init__(self):
            torch.manual_seed(1)
            y = torch.randint(0, 2, (8, len(CONCERNS))).float()
            y[:, 4:] = NAN                      # last two concerns never annotated
            self.batches = [(torch.randn(8, 3, 64, 64), torch.arange(8) % 4, y)]

        def __iter__(self):
            return iter(self.batches)

    metrics, _ = evaluate(build_model("simple"), _Loader(), "cpu")

    assert np.isfinite(metrics["concern_macro_f1"])
    assert metrics["concern_labeled_counts"][CONCERNS[0]] == 8
    assert metrics["concern_labeled_counts"][CONCERNS[4]] == 0
    assert metrics["concern_f1_per_class"][CONCERNS[4]] is None, \
        "an unannotated concern must report None, not a fabricated 0.0"
