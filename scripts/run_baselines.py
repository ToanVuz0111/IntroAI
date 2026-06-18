from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from course_recommender.baselines import run_baselines
from course_recommender.config import load_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full_smoke.yaml")
    print(run_baselines(load_config(parser.parse_args().config)).to_string(index=False))
