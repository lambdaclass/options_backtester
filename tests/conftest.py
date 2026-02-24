from __future__ import annotations

from pathlib import Path

import pytest


def pytest_collection_modifyitems(config, items):
    del config
    for item in items:
        path = Path(str(item.fspath)).as_posix()
        if "/tests/bench/" in path:
            item.add_marker(pytest.mark.bench)
