from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from course_recommender.ablation import (
    run_ablation,
    run_ablation_five_seeds,
    run_multi_seed,
    run_normalization_ablation,
    run_reward_sensitivity,
    run_similarity_sensitivity,
)
from course_recommender.config import load_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full_smoke.yaml")
    parser.add_argument("--extended", action="store_true")
    parser.add_argument("--five-seeds", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    print(run_ablation(config).to_string(index=False))
    print(run_similarity_sensitivity(config).to_string(index=False))
    if args.extended:
        print(run_reward_sensitivity(config).to_string(index=False))
        print(run_multi_seed(config, [42, 52, 62]).to_string(index=False))
        print(run_normalization_ablation(config).to_string(index=False))
    if args.five_seeds:
        _, summary = run_ablation_five_seeds(config)
        print(summary.to_string(index=False))
