from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from course_recommender.config import load_config
from course_recommender.rl_training import train_ppo_environment


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full_smoke.yaml")
    print(train_ppo_environment(load_config(parser.parse_args().config)))
