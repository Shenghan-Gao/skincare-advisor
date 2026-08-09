"""让测试在"未安装本项目"的情况下也能跑。

pytest 的控制台脚本不会把仓库根加进 sys.path(`python -m pytest` 会),
所以 `import app` 会失败。pyproject 已把 app 纳入安装包,这里再加一道保险,
这样组员即使忘了 `uv pip install -e .` 也能直接跑测试。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
