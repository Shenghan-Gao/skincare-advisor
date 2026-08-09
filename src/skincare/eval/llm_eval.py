"""[Deprecated] Use skincare.eval.run_eval instead.

An earlier evaluation entry point. Evaluation is now owned by member C and the single
entry point is:

    python -m skincare.eval.run_eval --self-test                    # self-test, no model
    python -m skincare.eval.run_eval --split data/processed/rl_test.jsonl

This file is kept only so that older imports keep working.
"""
from skincare.eval.run_eval import main, markdown_table, score_rows, self_test  # noqa: F401
