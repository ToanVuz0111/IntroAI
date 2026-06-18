from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .config import project_path
from .evaluation import ranking_metrics
from .recommendation import SimilarityModel, learner_profiles, load_bundle, recommend


def _evaluate_rankings(
    rankings: dict[str, list[str]],
    relevant: dict[str, set[str]],
    course_ids: list[str],
    embeddings: np.ndarray,
    k: int,
) -> dict[str, float]:
    rows = [ranking_metrics(rankings[user], relevant[user], k) for user in rankings]
    recommended = [course for values in rankings.values() for course in values[:k]]
    coverage = len(set(recommended)) / max(len(course_ids), 1)
    course_index = {value: index for index, value in enumerate(course_ids)}
    diversity_rows = []
    for values in rankings.values():
        indices = [course_index[value] for value in values[:k] if value in course_index]
        if len(indices) < 2:
            continue
        vectors = embeddings[indices]
        similarities = vectors @ vectors.T
        upper = similarities[np.triu_indices(len(indices), 1)]
        diversity_rows.append(float(1.0 - upper.mean()))
    return {
        **{
            name: float(np.mean([row[name] for row in rows])) if rows else 0.0
            for name in ("precision", "recall", "f1", "mrr", "ndcg")
        },
        "coverage": coverage,
        "diversity": float(np.mean(diversity_rows)) if diversity_rows else 0.0,
    }


def run_baselines(config: dict) -> pd.DataFrame:
    bundle = load_bundle(config)
    artifacts = project_path(config, config["artifacts_dir"])
    test = pd.read_csv(artifacts / "test.csv", dtype={"user_id": str, "course_id": str})
    relevant: dict[str, set[str]] = defaultdict(set)
    for row in test.itertuples(index=False):
        relevant[str(row.user_id)].add(str(row.course_id))
    users = list(relevant)[: int(config["evaluation"].get("max_users", 100))]
    k_values = config["evaluation"].get("k_values", [10])
    max_k = max(k_values)
    course_ids = bundle["course_ids"]
    popularity = (
        bundle["train"].groupby("course_id")["reward"].sum()
        .reindex(course_ids).fillna(0.0).to_numpy()
    )
    similarity_model = SimilarityModel(
        bundle["semantic_embeddings"],
        config["mahalanobis"],
        bundle["course_context"],
    )
    rankings_by_model: dict[str, dict[str, list[str]]] = {
        "Popularity": {},
        "BERT cosine": {},
        "Cosine": {},
        "Euclidean": {},
        "Mahalanobis": {},
        "Actor-Critic + PPO + fusion": {},
    }
    user_index = {value: index for index, value in enumerate(bundle["user_ids"])}
    rng = np.random.default_rng(config.get("seed", 42))
    rankings_by_model["Random"] = {}
    for user_id in users:
        index = user_index[user_id]
        seen = set(bundle["train"].loc[bundle["train"]["user_id"] == user_id, "course_id"].astype(str))
        mask = np.array([value not in seen for value in course_ids])
        def rank(scores):
            values = scores.copy()
            values[~mask] = -np.inf
            return [course_ids[i] for i in np.argsort(-values)[:max_k]]
        rankings_by_model["Popularity"][user_id] = rank(popularity)
        rankings_by_model["BERT cosine"][user_id] = rank(bundle["semantic_embeddings"] @ bundle["profiles"][index])
        cosine, _ = similarity_model.scores(
            bundle["profiles"][index],
            bundle["user_features"][index],
            metric="cosine",
        )
        rankings_by_model["Cosine"][user_id] = rank(cosine)
        euclidean, _ = similarity_model.scores(
            bundle["profiles"][index],
            bundle["user_features"][index],
            metric="euclidean",
        )
        rankings_by_model["Euclidean"][user_id] = rank(euclidean)
        mahalanobis, _ = similarity_model.scores(
            bundle["profiles"][index],
            bundle["user_features"][index],
        )
        rankings_by_model["Mahalanobis"][user_id] = rank(mahalanobis)
        random_scores = rng.random(len(course_ids))
        rankings_by_model["Random"][user_id] = rank(random_scores)
        rankings_by_model["Actor-Critic + PPO + fusion"][user_id] = (
            recommend(config, user_id, max_k, translate_output=False)["course_id"].astype(str).tolist()
        )
    rows = []
    for model, rankings in rankings_by_model.items():
        for k in k_values:
            rows.append({
                "model": model,
                "k": int(k),
                "users": len(rankings),
                **_evaluate_rankings(
                    rankings, relevant, course_ids, bundle["semantic_embeddings"], int(k),
                ),
            })
    output = pd.DataFrame(rows)
    output.to_csv(artifacts / "baseline_results.csv", index=False)
    return output
