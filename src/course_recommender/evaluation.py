from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .config import project_path
from .recommendation import recommend
from .utils import dump_json


def ranking_metrics(recommended: list[str], relevant: set[str], k: int) -> dict[str, float]:
    top = recommended[:k]
    hits = [course in relevant for course in top]
    precision = sum(hits) / max(k, 1)
    recall = sum(hits) / max(len(relevant), 1)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mrr = next((1.0 / rank for rank, hit in enumerate(hits, 1) if hit), 0.0)
    dcg = sum((1.0 if hit else 0.0) / np.log2(rank + 1) for rank, hit in enumerate(hits, 1))
    ideal = sum(1.0 / np.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mrr": mrr,
        "ndcg": dcg / ideal if ideal else 0.0,
    }


def evaluate(config: dict) -> pd.DataFrame:
    artifacts = project_path(config, config["artifacts_dir"])
    test = pd.read_csv(artifacts / "test.csv", dtype={"user_id": str, "course_id": str})
    relevant: dict[str, set[str]] = defaultdict(set)
    for row in test.itertuples(index=False):
        relevant[str(row.user_id)].add(str(row.course_id))
    k_values = config["evaluation"].get("k_values", [config["recommendation"].get("top_k", 10)])
    max_k = max(k_values)
    max_users = int(config["evaluation"].get("max_users", 100))
    recommendations = {}
    for user_id in list(relevant)[:max_users]:
        recommendations[user_id] = recommend(
            config, user_id, max_k, translate_output=False,
        )["course_id"].astype(str).tolist()

    rows = []
    for k in k_values:
        metrics = [
            ranking_metrics(recommendations[user_id], relevant[user_id], int(k))
            for user_id in recommendations
        ]
        rows.append({
            "model": "Actor-Critic + PPO + BERT + similarity fusion",
            "k": int(k),
            "users": len(metrics),
            **{
                name: float(np.mean([row[name] for row in metrics])) if metrics else 0.0
                for name in ("precision", "recall", "f1", "mrr", "ndcg")
            },
        })
    output = pd.DataFrame(rows)
    output.to_csv(artifacts / "evaluation.csv", index=False)
    dump_json(artifacts / "evaluation_summary.json", rows)
    return output
