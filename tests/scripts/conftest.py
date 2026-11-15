"""Pytest config for tests/scripts/.

Adds the repo root to sys.path so ``from scripts.improvement_workers...``
resolves. The repo's main package is ``src`` (see pyproject ``[tool.hatch.build.targets.wheel]``);
``scripts`` is not installed but is imported as a top-level namespace by the
register-workers scripts at runtime via the same sys.path insertion done in
their own ``__main__`` blocks.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
