from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from course_recommender.data import build_splits, split_interactions
from course_recommender.evaluation import ranking_metrics
from course_recommender.features import hash_embeddings
from course_recommender.environment import CourseRecommendationEnv
from course_recommender.models import Actor, CatalogActor, Critic, make_states
from course_recommender.ppo import RolloutBuffer
from course_recommender.recommendation import SimilarityModel
from course_recommender.translation import TopKTranslator
from course_recommender.utils import anonymize_course_id, anonymize_user_id, repair_mojibake


def test_state_dimension_and_models() -> None:
    states = make_states(torch.zeros(10), torch.zeros((4, 768)))
    assert states.shape == (4, 778)
    assert Actor()(states).shape == (4,)
    assert Critic()(states).shape == (4,)
    catalog_logits = CatalogActor(7)(torch.zeros((2, 778)))
    assert catalog_logits.shape == (2, 7)


def test_hash_embedding_shape_and_norm() -> None:
    values = hash_embeddings(["machine learning", ""])
    assert values.shape == (2, 768)
    assert np.isclose(np.linalg.norm(values[0]), 1.0)


def test_temporal_split_has_no_future_leakage() -> None:
    frame = pd.DataFrame({
        "user_id": ["u"] * 4,
        "course_id": ["a", "b", "c", "d"],
        "timestamp": pd.date_range("2025-01-01", periods=4, tz="UTC"),
    })
    train, _, test = split_interactions(frame, "temporal", 42)
    assert test.empty or train["timestamp"].max() <= test["timestamp"].min()


def test_user_hash_is_stable_and_anonymous() -> None:
    assert anonymize_user_id("U_1", "salt") == anonymize_user_id("U_1", "salt")
    assert "U_1" not in anonymize_user_id("U_1", "salt")
    assert anonymize_course_id("C_1", "salt").startswith("c_")
    assert "C_1" not in anonymize_course_id("C_1", "salt")


def test_metrics() -> None:
    result = ranking_metrics(["a", "b"], {"b"}, 2)
    assert result["precision"] == 0.5
    assert result["mrr"] == 0.5


def test_translation_and_encoding_repair() -> None:
    corrupted = "è‡ªç„¶ç¾å®³"
    repaired = repair_mojibake(corrupted)
    assert repaired == "自然灾害"
    assert TopKTranslator().translate(repaired) == "Natural Disasters"


def test_rollout_advantages() -> None:
    buffer = RolloutBuffer(
        rewards=[1.0, 0.5],
        values=[0.2, 0.1],
        dones=[False, True],
    )
    advantages, returns = buffer.advantages(0.0, gamma=0.99, gae_lambda=0.95)
    assert advantages.shape == (2,)
    assert returns.shape == (2,)
    assert torch.isfinite(advantages).all()


def test_environment_reset_step_and_mask() -> None:
    interactions = pd.DataFrame({
        "user_id": ["u", "u", "u"],
        "course_id": ["a", "b", "c"],
        "timestamp": pd.date_range("2025-01-01", periods=3, tz="UTC"),
        "reward": [0.8, 0.7, 0.9],
        "completion_rate": [0.8, 0.7, 0.9],
        "quiz_score": [0.8, 0.7, 0.9],
        "engagement_time": [0.8, 0.7, 0.9],
    })
    embeddings = np.eye(3, 768, dtype=np.float32)
    env = CourseRecommendationEnv(
        interactions,
        np.zeros((1, 10), dtype=np.float32),
        ["u"],
        ["a", "b", "c"],
        embeddings,
        max_steps=2,
        seed=1,
    )
    observation, info = env.reset(options={"user_id": "u"})
    assert observation["state"].shape == (778,)
    assert info["user_id"] == "u"
    valid_action = int(np.flatnonzero(observation["action_mask"])[0])
    next_observation, reward, terminated, truncated, _ = env.step(valid_action)
    assert np.isfinite(reward)
    assert not next_observation["action_mask"][valid_action]
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_combined_similarity_supports_euclidean_and_mahalanobis() -> None:
    rng = np.random.default_rng(42)
    embeddings = rng.normal(size=(20, 768)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    context = rng.random((20, 10), dtype=np.float32)
    scorer = SimilarityModel(
        embeddings,
        {"pca_components": 8, "n_clusters": 3, "temperature": 1.0},
        context,
    )
    for metric in ("euclidean", "cosine", "mahalanobis"):
        similarity, affinity = scorer.scores(embeddings[0], context[0], metric)
        assert similarity.shape == (20,)
        assert np.isfinite(similarity).all()
        assert ((similarity >= 0.0) & (similarity <= 1.0)).all()
        assert set(np.unique(affinity)).issubset({0.0, 1.0})


def test_train_only_imputation_and_scaling() -> None:
    root = ROOT / "artifacts" / "test_preprocess"
    processed = root / "processed"
    artifacts = root / "artifacts"
    processed.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "course_id": ["c1", "c2"],
        "source_course_id": ["raw1", "raw2"],
        "title": ["A", "B"],
        "description": ["", ""],
        "tags": ["", ""],
        "category": ["x", "x"],
        "difficulty": ["beginner", "advanced"],
        "source_dataset": ["test", "test"],
        "total_videos": [1, 2],
        "text": ["A", "B"],
    }).to_csv(processed / "courses.csv", index=False)
    pd.DataFrame({
        "user_id": ["u1", "u2"],
        "age": [np.nan, np.nan],
        "gender": ["unknown", "unknown"],
        "education_level": ["unknown", "unknown"],
        "source_dataset": ["test", "test"],
    }).to_csv(processed / "users.csv", index=False)
    rows = []
    for index in range(20):
        rows.append({
            "interaction_id": f"i{index}",
            "user_id": f"u{index % 2 + 1}",
            "course_id": f"c{index % 2 + 1}",
            "timestamp": f"2025-01-{index + 1:02d}",
            "rating": np.nan if index < 14 else 5.0,
            "clicked": 1,
            "video_views": index,
            "completion_rate": 0.5,
            "quiz_score": np.nan if index < 14 else 0.9,
            "engagement_time": 0.5,
            "access_frequency": index + 1,
            "session_duration": index + 1,
            "device_type": "unknown",
            "interaction_type": "test",
            "source_dataset": "test",
            "is_derived": False,
            "reward": 0.5,
        })
    pd.DataFrame(rows).to_csv(processed / "interactions.csv", index=False)
    config = {
        "project_root": str(root),
        "seed": 42,
        "artifacts_dir": "artifacts",
        "data": {
            "processed_dir": "processed",
            "split_strategy": "paper_random",
            "outlier_filter": {"enabled": False},
        },
    }
    build_splits(config)
    meta = __import__("json").loads((artifacts / "split_summary.json").read_text())
    assert meta["split_strategy"] == "paper_random"
    assert "quiz_score" in meta["mean_imputation_train_only"]
    train = pd.read_csv(artifacts / "train.csv")
    assert not train["quiz_score"].isna().any()
    assert train["difficulty_score"].between(0.0, 1.0).all()
