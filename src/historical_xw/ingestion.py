from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable


class HistoricalCache:
    """Content-addressed local cache shared by FastF1/OpenF1/Jolpica adapters."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def get_or_fetch(self, source: str, request: dict[str, Any], fetch: Callable[[], Any]) -> Path:
        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:20]
        path = self.root / source / f"{key}.json"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = fetch()
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path


class SourceRegistry:
    supported = ("fastf1", "openf1", "jolpica", "manual_technical_updates")

    @classmethod
    def validate(cls, source: str) -> None:
        if source not in cls.supported:
            raise ValueError(f"Unsupported source {source!r}; choose one of {cls.supported}")

