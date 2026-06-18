from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config["_config_path"] = str(config_path.resolve())
    return config


def project_path(config: dict[str, Any], value: str) -> Path:
    root = Path(config.get("project_root", "."))
    return (root / value).resolve()
