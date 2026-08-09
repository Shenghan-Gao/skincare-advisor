"""Let the tests run even when this project has not been installed.

The pytest console script does not put the repository root on sys.path (`python -m pytest`
does), so `import app` would fail. pyproject already ships app as part of the installed
package; this is a second line of defence so that a teammate who forgot to run
`uv pip install -e .` can still run the tests directly.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
