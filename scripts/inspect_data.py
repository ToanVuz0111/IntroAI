from __future__ import annotations

import json
from pathlib import Path


def first_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.loads(next(handle))


if __name__ == "__main__":
    root = Path("dataset/MOOCCube")
    files = {
        "course": root / "entities/course.json",
        "user": root / "entities/user.json",
        "user-course": root / "relations/user-course.json",
    }
    for name, path in files.items():
        size_mb = path.stat().st_size / 1024 / 1024
        if path.suffix == ".json" and name != "user-course":
            print(f"{name}: {size_mb:.2f} MB, keys={list(first_json(path))}")
        else:
            print(f"{name}: {size_mb:.2f} MB")
