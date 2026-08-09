"""Handoff acceptance script -- a teammate's work counts as delivered only once this passes;
do not merge otherwise.

    python scripts/verify_handoff.py vision models/vision/transfer_resnet18_lr3e4.pt
    python scripts/verify_handoff.py rag

What it checks is the "contract", not how good the results are: whether the mainline code
can load the file, whether the output shapes are right, whether the interface is usable.
How good the results are is what the metrics table is for -- a separate question.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

OK, BAD = "  [ok]", "  [FAIL]"


def verify_vision(ckpt_path: str) -> bool:
    import torch
    from app.schemas import SkinAnalysis
    from skincare.vision.infer import SkinClassifier

    print(f"verifying vision checkpoint: {ckpt_path}")
    p = Path(ckpt_path)
    if not p.exists():
        print(BAD, "file not found"); return False
    print(OK, f"file exists ({p.stat().st_size / 1e6:.1f} MB)")

    ck = torch.load(p, map_location="cpu")
    for key in ("state_dict", "kind", "metrics"):
        if key not in ck:
            print(BAD, f"checkpoint missing required key '{key}'"); return False
    print(OK, f"keys present; kind={ck['kind']}")
    print(OK, f"metrics reported: {json.dumps(ck['metrics'])}")

    try:
        clf = SkinClassifier(str(p), device="cpu")
    except Exception as e:
        print(BAD, f"SkinClassifier could not load it: {e}"); return False
    print(OK, "loads through SkinClassifier")

    # Run a forward pass on a dummy image to confirm the output meets the SkinAnalysis contract
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (300, 300), (180, 140, 130)).save(buf, format="JPEG")
    out = clf.predict_bytes(buf.getvalue())
    if not isinstance(out, SkinAnalysis):
        print(BAD, "predict_bytes did not return SkinAnalysis"); return False
    if len(out.concerns) != 6:
        print(BAD, f"expected 6 concern scores, got {len(out.concerns)}"); return False
    print(OK, f"inference contract holds -> {out.skin_type.value} "
              f"({out.skin_type_confidence:.2f}), {len(out.concerns)} concerns")
    print("\nPASS -- ready to deliver; hand Anna the .pt together with the metrics table\n")
    return True


def verify_rag() -> bool:
    from app.schemas import SkinAnalysis, UserProfile
    from skincare.rag.retrieve import Retriever

    print("verifying RAG artifacts")
    from skincare.config import PROCESSED
    for f in ["index/chunks.faiss", "index/chunks_meta.parquet", "products.parquet"]:
        if not (PROCESSED / f).exists():
            print(BAD, f"missing {PROCESSED / f}"); return False
    print(OK, "all artifact files present")

    r = Retriever()
    res = r.search(UserProfile(query="oily skin with acne and large pores", budget_usd=50),
                   SkinAnalysis(skin_type="oily", skin_type_confidence=0.9,
                                concerns=[{"concern": "acne", "score": 0.8}]), top_k=3)
    if not res.products:
        print(BAD, "query returned no products"); return False
    if not res.evidence:
        print(BAD, "query returned no evidence"); return False
    ids = [e.evidence_id for e in res.evidence]
    if len(ids) != len(set(ids)):
        print(BAD, "evidence_id not unique -- the reward function will misjudge"); return False
    print(OK, f"{len(res.products)} products, {len(res.evidence)} evidence, ids unique")
    print(OK, f"sample evidence_id: {ids[0]}")
    print("\nPASS -- the retrieval pipeline works\n")
    return True


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "vision":
        ok = verify_vision(sys.argv[2] if len(sys.argv) > 2 else "models/vision/best.pt")
    elif mode == "rag":
        ok = verify_rag()
    else:
        print(__doc__); ok = False
    sys.exit(0 if ok else 1)
