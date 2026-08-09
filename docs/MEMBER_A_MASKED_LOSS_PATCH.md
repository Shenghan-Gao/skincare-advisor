# Required B-side Patch for Partial Concern Labels

Member A intentionally leaves unknown concern labels blank/NaN. This avoids treating "not annotated" as "negative".

Do **not** apply this without B/Anna's agreement because `src/skincare/vision/**` is B/Anna-owned. B should make the equivalent change before training.

```diff
--- a/src/skincare/vision/data.py
+++ b/src/skincare/vision/data.py
@@
-        y_concern = torch.tensor([float(row.get(c, 0)) for c in CONCERNS])
+        values = [float(row[c]) if pd.notna(row[c]) else -1.0 for c in CONCERNS]
+        y_concern = torch.tensor(values, dtype=torch.float32)
```

```diff
--- a/src/skincare/vision/model.py
+++ b/src/skincare/vision/model.py
@@
 def multitask_loss(type_logits, concern_logits, y_type, y_concern, w: float = 1.0):
     ce = nn.functional.cross_entropy(type_logits, y_type)
-    bce = nn.functional.binary_cross_entropy_with_logits(concern_logits, y_concern)
+    mask = y_concern >= 0
+    safe_target = y_concern.clamp_min(0)
+    raw = nn.functional.binary_cross_entropy_with_logits(
+        concern_logits, safe_target, reduction="none"
+    )
+    bce = (raw * mask).sum() / mask.sum().clamp_min(1)
     return ce + w * bce, ce.item(), bce.item()
```
