"""[已废弃] 请用 skincare.eval.run_eval

早期版本的评估入口。现在评估体系由组员 C 拥有,统一入口是:

    python -m skincare.eval.run_eval --self-test                    # 无模型自检
    python -m skincare.eval.run_eval --split data/processed/rl_test.jsonl

保留此文件仅为兼容旧引用。
"""
from skincare.eval.run_eval import main, markdown_table, score_rows, self_test  # noqa: F401
