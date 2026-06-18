from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from course_recommender.bert_training import fine_tune_bert
from course_recommender.baselines import run_baselines
from course_recommender.config import load_config
from course_recommender.data import build_splits, prepare_mooccube
from course_recommender.evaluation import evaluate
from course_recommender.eda import run_eda
from course_recommender.efficiency import profile_efficiency
from course_recommender.features import build_features
from course_recommender.rl_training import train_ppo_environment
from course_recommender.training import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full_smoke.yaml")
    parser.add_argument("--skip-finetune", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    print("DATA", prepare_mooccube(config))
    print("SPLIT", build_splits(config))
    if (
        config["embedding"].get("backend") == "bert"
        and config.get("bert_finetuning", {}).get("enabled")
        and not args.skip_finetune
    ):
        print("BERT", fine_tune_bert(config))
    print("FEATURES", build_features(config))
    print("TRAIN", train(config))
    if config.get("environment"):
        print("PPO", train_ppo_environment(config))
    print("EVALUATION")
    print(evaluate(config).to_string(index=False))
    print("BASELINES")
    print(run_baselines(config).to_string(index=False))
    print("EDA", run_eda(config))
    print("EFFICIENCY", profile_efficiency(config, int(config.get("efficiency", {}).get("sessions", 100))))
