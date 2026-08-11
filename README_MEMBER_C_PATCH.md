# Member C GitHub patch v3

Use this version instead of v1/v2.

Files:
- `src/skincare/augment/diffusion_aug.py`
- `src/skincare/eval/rag_eval.py`
- `tests/test_member_c.py`

Important workflow detail fixed in v3:
- diffusion mode generates pilot images + `metadata.csv` only;
- it does **not** automatically add generated faces to training;
- manual QC must set `accepted=1/0` first;
- `build-synthetic-csv` only admits accepted rows;
- the reported final fallback remains `combination` oversampling +200.

Validated against the project repository and the real handoff files available in the conversation:
- repository tests after overlay: 44 passed;
- real vision train: 2,718 -> 2,918 rows and combination 230 -> 430;
- all six concern NaN counts: 2,373 -> 2,573, matching the reported fallback behavior;
- blinded RAG re-evaluation: MiniLM Precision@3 = 0.791666..., MPNet = 0.625.

The Stable Diffusion model id follows the current Hugging Face Diffusers img2img docs:
`stable-diffusion-v1-5/stable-diffusion-v1-5`.

Do not commit raw/processed datasets, generated face datasets, checkpoints, API keys,
or private blind-source mappings.

Note: `src/skincare/eval/judge.py` is a separate remaining C-owned task and is not
included in this patch.
