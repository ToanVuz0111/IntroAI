from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

from course_recommender.config import load_config
from course_recommender.recommendation import recommend
from course_recommender.utils import load_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fast_demo.yaml")
    parser.add_argument("--user-id")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    config = load_config(args.config)
    artifacts = ROOT / config["artifacts_dir"]
    user_id = args.user_id or str(load_json(artifacts / "user_ids.json")[0])
    print(recommend(config, user_id, args.top_k).to_string(index=False))
