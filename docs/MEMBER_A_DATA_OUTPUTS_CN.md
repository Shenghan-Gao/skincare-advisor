# Member A 已清洗数据交接包

这个包是根据你实际上传的 4 个 Kaggle ZIP 跑出来的，不是模拟数据。

## vision/

- `vision_train.csv`: 2,718 行
- `vision_val.csv`: 584 行
- `vision_test.csv`: 579 行
- `class_distribution.md`: 清洗审计 + 类别分布
- `vision_concern_aux.csv`: 60 张 acne/redness 辅助图片标签；暂时不要直接并入主训练集

注意：主 Vision CSV 的 concern 列有空值是**故意的**。原数据只有 200 条详细 concern 标注，不能把“未标注”伪造成 0。B 在训练 concern head 前必须做 masked BCE。具体 patch 在代码包的 `docs/MEMBER_A_MASKED_LOSS_PATCH.md`。

## sephora/

- `products_clean.csv`: 2,282 个清洗后的 skincare products（便于人工检查/共享）
- `reviews_clean_top20.csv`: 38,525 条最终保留 review
- `chunks_clean.csv`: 43,089 条 RAG evidence chunks

正式项目契约要求的是 `products.parquet` 和 `chunks.parquet`。代码包中的 `python -m skincare.rag.ingest` 会直接从原始 Sephora CSV 生成这两个 parquet。`pyarrow` 已加入 `rag` 依赖。

## audit/

- `sephora_cleaning_audit.json`: Sephora 各步骤真实行数
- `vision_final_audit.json`: Vision 去重、冲突、split 真实统计
- `vision_audit_labels.csv`: Vision 详细审计表
- `cross_pairs.jpg`: 部分跨类别近重复/冲突图片的检查图

## 原始数据如何放置

原始数据不要提交 GitHub。需要本地复现时：

```text
data/raw/sephora/
  product_info.csv
  reviews_*.csv

data/raw/vision/skin_type_classification_dataset/
  train/
  valid/
  test/
  skinalaysis_labeling_train1.xlsx
  skinanalysis_valid1.xlsx

data/raw/vision/skin_defects/
  files/
  skin_defects.csv
```

第一个只有 45 张、没有目标标签的 Facial Skin Condition Dataset 没有并入主 CNN 数据，因为不能人为编标签。
