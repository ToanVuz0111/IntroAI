from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, pearsonr

from .config import project_path
from .utils import dump_json


def run_eda(config: dict) -> dict:
    artifacts = project_path(config, config["artifacts_dir"])
    train = pd.read_csv(artifacts / "train.csv")
    validation = pd.read_csv(artifacts / "val.csv")
    test = pd.read_csv(artifacts / "test.csv")
    interactions = pd.concat(
        [
            train.assign(split="train"),
            validation.assign(split="validation"),
            test.assign(split="test"),
        ],
        ignore_index=True,
    )
    numeric = [
        "reward", "completion_rate", "quiz_score", "engagement_time",
        "session_duration", "access_frequency", "video_views",
    ]
    available = [column for column in numeric if column in interactions]
    distribution = interactions[available].describe(
        percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99],
    ).T
    distribution.to_csv(artifacts / "eda_distributions.csv")

    correlations = []
    for column in available:
        if column == "reward":
            continue
        clean = interactions[[column, "reward"]].dropna()
        if len(clean) >= 3 and clean[column].nunique() > 1:
            coefficient, p_value = pearsonr(clean[column], clean["reward"])
            correlations.append({
                "feature": column,
                "pearson_r": float(coefficient),
                "p_value": float(p_value),
                "samples": len(clean),
            })
    pd.DataFrame(correlations).to_csv(artifacts / "eda_correlations.csv", index=False)

    outliers = []
    for column in ["session_duration", "access_frequency", "engagement_time"]:
        if column not in interactions:
            continue
        q1, q3 = interactions[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 3.0 * iqr, q3 + 3.0 * iqr
        mask = ~interactions[column].between(lower, upper)
        outliers.append({
            "feature": column,
            "lower_bound": float(lower),
            "upper_bound": float(upper),
            "outlier_count": int(mask.sum()),
            "outlier_ratio": float(mask.mean()),
        })
    pd.DataFrame(outliers).to_csv(artifacts / "eda_outliers.csv", index=False)

    difficulty_result = {
        "available": False,
        "groups": 0,
        "anova_f": None,
        "anova_p": None,
    }
    if "difficulty" in interactions:
        known = interactions[
            interactions["difficulty"].fillna("unknown").astype(str).str.lower() != "unknown"
        ]
        groups = [
            group["reward"].dropna().to_numpy()
            for _, group in known.groupby("difficulty")
            if len(group) >= 2
        ]
        if len(groups) >= 2:
            statistic, p_value = f_oneway(*groups)
            difficulty_result = {
                "available": True,
                "groups": len(groups),
                "anova_f": float(statistic),
                "anova_p": float(p_value),
            }
            known.groupby("difficulty")["reward"].agg(
                ["count", "mean", "std"],
            ).to_csv(artifacts / "eda_difficulty.csv")

    summary = {
        "rows": len(interactions),
        "users": int(interactions["user_id"].nunique()),
        "courses": int(interactions["course_id"].nunique()),
        "reward_mean": float(interactions["reward"].mean()),
        "reward_std": float(interactions["reward"].std()),
        "interaction_duration_mean": float(interactions["session_duration"].mean()),
        "interaction_duration_std": float(interactions["session_duration"].std()),
        "strongest_reward_correlation": (
            max(correlations, key=lambda row: abs(row["pearson_r"]))
            if correlations
            else None
        ),
        "difficulty_analysis": difficulty_result,
        "difficulty_note": (
            "No observed difficulty labels are available in public MOOCCube."
            if not difficulty_result["available"]
            else ""
        ),
    }
    dump_json(artifacts / "eda_summary.json", summary)
    return summary
