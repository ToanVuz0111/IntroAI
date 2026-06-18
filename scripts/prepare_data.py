from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from course_recommender.config import load_config
from course_recommender.data import build_splits, prepare_mooccube


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fast_demo.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    print(prepare_mooccube(config))
    print(build_splits(config))


if __name__ == "__main__":
    main()
