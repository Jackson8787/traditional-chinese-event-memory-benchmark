from __future__ import annotations

from pathlib import Path


SRC_PACKAGE = Path(__file__).resolve().parents[1] / "src" / "event_memory"
if SRC_PACKAGE.exists():
    __path__.insert(0, str(SRC_PACKAGE))  # type: ignore[name-defined]
