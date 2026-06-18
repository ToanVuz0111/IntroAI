from __future__ import annotations

import copy

import pandas as pd

from .baselines import run_baselines
from .config import project_path
from .evaluation import evaluate
from .rl_training import train_ppo_environment
from .training import train


def run_ablation(config: dict) -> pd.DataFrame:
    variants = ablation_variants()
    rows = []
    for name, changes in variants.items():
        variant = copy.deepcopy(config)
        for path, value in changes.items():
            section, key = path.split(".")
            variant[section][key] = value
        result = run_baselines(variant)
        selected = result[result["model"] == "Actor-Critic + PPO + fusion"].copy()
        selected.insert(0, "variant", name)
        rows.append(selected)
    output = pd.concat(rows, ignore_index=True)
    artifacts = project_path(config, config["artifacts_dir"])
    output.to_csv(artifacts / "ablation_results.csv", index=False)
    return output


def ablation_variants() -> dict[str, dict[str, object]]:
    return {
        "full": {},
        "without_semantic": {"recommendation.semantic_weight": 0.0},
        "without_mahalanobis": {"mahalanobis.logit_fusion_alpha": 0.0},
        "without_cluster": {"mahalanobis.logit_fusion_beta": 0.0},
        "without_ppo": {"recommendation.use_ppo": False},
        "without_bert": {
            "recommendation.disable_bert": True,
            "recommendation.semantic_weight": 0.0,
            "mahalanobis.logit_fusion_alpha": 0.0,
            "mahalanobis.logit_fusion_beta": 0.0,
        },
        "candidate_set": {"recommendation.action_space": "candidate_set"},
    }


def run_ablation_five_seeds(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    seeds = list(config.get("evaluation", {}).get("seeds", [42, 52, 62, 72, 82]))
    raw_rows = []
    for seed in seeds:
        seeded = copy.deepcopy(config)
        seeded["seed"] = int(seed)
        train(seeded)
        train_ppo_environment(seeded)
        for name, changes in ablation_variants().items():
            variant = copy.deepcopy(seeded)
            for path, value in changes.items():
                section, key = path.split(".")
                variant[section][key] = value
            result = run_baselines(variant)
            selected = result[result["model"] == "Actor-Critic + PPO + fusion"].copy()
            selected.insert(0, "seed", seed)
            selected.insert(0, "variant", name)
            raw_rows.append(selected)
    original = copy.deepcopy(config)
    train(original)
    train_ppo_environment(original)
    raw = pd.concat(raw_rows, ignore_index=True)
    metrics = ["precision", "recall", "f1", "mrr", "ndcg", "coverage", "diversity"]
    summary = raw.groupby(["variant", "k"], as_index=False)[metrics].agg(["mean", "std"])
    summary.columns = [
        "_".join([str(part) for part in column if str(part)])
        for column in summary.columns.to_flat_index()
    ]
    artifacts = project_path(config, config["artifacts_dir"])
    raw.to_csv(artifacts / "ablation_five_seed_raw.csv", index=False)
    summary.to_csv(artifacts / "ablation_five_seed_summary.csv", index=False)
    return raw, summary


def run_similarity_sensitivity(config: dict) -> pd.DataFrame:
    result = run_baselines(config)
    output = result[result["model"].isin(["Euclidean", "Cosine", "Mahalanobis"])].copy()
    artifacts = project_path(config, config["artifacts_dir"])
    output.to_csv(artifacts / "similarity_sensitivity.csv", index=False)
    return output


def run_reward_sensitivity(config: dict) -> pd.DataFrame:
    artifacts = project_path(config, config["artifacts_dir"])
    source = pd.read_csv(artifacts / "train.csv")
    weights = [
        (0.5, 0.3, 0.2),
        (0.3, 0.5, 0.2),
        (0.4, 0.4, 0.2),
        (0.4, 0.3, 0.3),
        (0.3, 0.4, 0.3),
    ]
    rows = []
    original = source["reward"].copy()
    try:
        for completion, quiz, engagement in weights:
            source["reward"] = (
                completion * source["completion_rate"]
                + quiz * source["quiz_score"]
                + engagement * source["engagement_time"]
            )
            source.to_csv(artifacts / "train.csv", index=False)
            train_ppo_environment(config)
            metrics = evaluate(config)
            selected = metrics[metrics["k"] == 10].iloc[0].to_dict()
            rows.append({
                "completion_weight": completion,
                "quiz_weight": quiz,
                "engagement_weight": engagement,
                **selected,
            })
    finally:
        source["reward"] = original
        source.to_csv(artifacts / "train.csv", index=False)
        train_ppo_environment(config)
    output = pd.DataFrame(rows)
    output.to_csv(artifacts / "reward_sensitivity.csv", index=False)
    return output


def run_multi_seed(config: dict, seeds: list[int]) -> pd.DataFrame:
    rows = []
    original_seed = config.get("seed", 42)
    for seed in seeds:
        variant = copy.deepcopy(config)
        variant["seed"] = seed
        train_ppo_environment(variant)
        metrics = evaluate(variant)
        metrics.insert(0, "seed", seed)
        rows.append(metrics)
    config["seed"] = original_seed
    train_ppo_environment(config)
    output = pd.concat(rows, ignore_index=True)
    artifacts = project_path(config, config["artifacts_dir"])
    output.to_csv(artifacts / "multi_seed_results.csv", index=False)
    return output


def run_normalization_ablation(config: dict) -> pd.DataFrame:
    variants = {
        "normalization_full": {},
        "without_reward_normalization": {"normalize_rewards": False},
        "without_advantage_normalization": {"normalize_advantages": False},
    }
    rows = []
    original_environment = copy.deepcopy(config.get("environment", {}))
    for name, changes in variants.items():
        variant = copy.deepcopy(config)
        variant.setdefault("environment", {}).update(changes)
        train_ppo_environment(variant)
        metrics = evaluate(variant)
        metrics.insert(0, "variant", name)
        rows.append(metrics)
    config["environment"] = original_environment
    train_ppo_environment(config)
    output = pd.concat(rows, ignore_index=True)
    artifacts = project_path(config, config["artifacts_dir"])
    output.to_csv(artifacts / "normalization_ablation.csv", index=False)
    return output
