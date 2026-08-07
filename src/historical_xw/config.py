from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class Settings(BaseModel):
    raw: dict[str, Any]

    @property
    def data_dir(self) -> Path:
        return Path(self.raw["project"]["data_directory"])

    @property
    def seed(self) -> int:
        return int(self.raw["project"]["random_seed"])


def load_settings(path: Path) -> Settings:
    return Settings(raw=yaml.safe_load(path.read_text(encoding="utf-8")))
