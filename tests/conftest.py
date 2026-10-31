from pathlib import Path

import pytest

from ragsvc.index import HybridIndex

KB = Path(__file__).resolve().parent.parent / "kb"


@pytest.fixture(scope="session")
def kb_index():
    idx = HybridIndex()
    for p in sorted(KB.glob("*.md")):
        idx.upsert(p.stem, p.read_text(encoding="utf-8"))
    return idx
