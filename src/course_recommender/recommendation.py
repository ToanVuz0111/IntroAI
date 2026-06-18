from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.covariance import LedoitWolf
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .config import project_path
from .models import Actor, CatalogActor, make_states
from .translation import TopKTranslator
from .utils import ensure_dir, load_json


_SIMILARITY_CACHE: dict[str, "SimilarityModel"] = {}
_BUNDLE_CACHE: dict[str, dict] = {}


def build_course_context(
    train: pd.DataFrame,
    course_ids: list[str],
) -> np.ndarray:
    columns = [
        "completion_rate", "quiz_score", "engagement_time", "rating",
        "clicked", "video_views", "access_frequency", "session_duration",
        "difficulty_score", "reward",
    ]
    available = [column for column in columns if column in train.columns]
    grouped = train.groupby("course_id")[available].mean()
    context = grouped.reindex(course_ids).fillna(0.0)
    for column in columns:
        if column not in context:
            context[column] = 0.0
    values = context[columns].to_numpy(dtype=np.float32)
    minimum = values.min(axis=0)
    maximum = values.max(axis=0)
    denominator = np.where(maximum > minimum, maximum - minimum, 1.0)
    return np.clip((values - minimum) / denominator, 0.0, 1.0).astype(np.float32)


def learner_profiles(
    train: pd.DataFrame,
    user_ids: list[str],
    course_ids: list[str],
    embeddings: np.ndarray,
) -> np.ndarray:
    users = {value: index for index, value in enumerate(user_ids)}
    courses = {value: index for index, value in enumerate(course_ids)}
    profiles = np.zeros((len(user_ids), embeddings.shape[1]), dtype=np.float32)
    for user_id, group in train.groupby("user_id"):
        vectors, weights = [], []
        for row in group.itertuples(index=False):
            course_index = courses.get(str(row.course_id))
            if course_index is not None:
                vectors.append(embeddings[course_index])
                weights.append(max(float(row.reward), 1e-3))
        if vectors and str(user_id) in users:
            profile = np.average(np.vstack(vectors), axis=0, weights=weights)
            profiles[users[str(user_id)]] = profile / max(np.linalg.norm(profile), 1e-8)
    return profiles


class SimilarityModel:
    def __init__(
        self,
        embeddings: np.ndarray,
        config: dict,
        course_context: np.ndarray | None = None,
    ) -> None:
        self.embeddings = embeddings
        components = min(
            int(config.get("pca_components", 64)),
            embeddings.shape[0] - 1,
            embeddings.shape[1],
        )
        self.scaler = StandardScaler().fit(embeddings)
        standardized = self.scaler.transform(embeddings)
        self.pca = PCA(n_components=max(components, 2), whiten=True, random_state=42).fit(standardized)
        reduced_semantic = self.pca.transform(standardized).astype(np.float32)
        if course_context is None:
            course_context = np.zeros((len(embeddings), 10), dtype=np.float32)
        self.context_scaler = StandardScaler().fit(course_context)
        reduced_context = self.context_scaler.transform(course_context).astype(np.float32)
        context_weight = float(config.get("context_weight", 1.0))
        combined = np.concatenate(
            [reduced_semantic, context_weight * reduced_context],
            axis=1,
        )
        self.reduced = combined
        self.precision = LedoitWolf().fit(combined).precision_.astype(np.float32)
        clusters = min(config.get("n_clusters", 8), len(embeddings))
        self.labels = KMeans(n_clusters=max(clusters, 1), random_state=42, n_init=10).fit_predict(combined)
        self.temperature = float(config.get("temperature", 1.0))
        self.context_weight = context_weight

    def distances(
        self,
        profile: np.ndarray,
        user_context: np.ndarray | None = None,
        metric: str = "mahalanobis",
    ) -> np.ndarray:
        reduced_profile = self.pca.transform(self.scaler.transform(profile.reshape(1, -1)))[0]
        if user_context is None:
            user_context = np.zeros(10, dtype=np.float32)
        reduced_context = self.context_scaler.transform(user_context.reshape(1, -1))[0]
        combined_profile = np.concatenate([
            reduced_profile,
            self.context_weight * reduced_context,
        ])
        difference = self.reduced - combined_profile.reshape(1, -1)
        if metric == "euclidean":
            return np.linalg.norm(difference, axis=1)
        if metric == "cosine":
            numerator = self.reduced @ combined_profile
            denominator = (
                np.linalg.norm(self.reduced, axis=1)
                * max(np.linalg.norm(combined_profile), 1e-8)
            )
            return 1.0 - numerator / np.maximum(denominator, 1e-8)
        return np.sqrt(np.maximum(
            np.einsum("ij,jk,ik->i", difference, self.precision, difference),
            0.0,
        ))

    def scores(
        self,
        profile: np.ndarray,
        user_context: np.ndarray | None = None,
        metric: str = "mahalanobis",
    ) -> tuple[np.ndarray, np.ndarray]:
        distance = self.distances(profile, user_context, metric)
        scale = np.median(distance[distance > 0]) if np.any(distance > 0) else 1.0
        similarity = np.exp(-distance / max(self.temperature * scale, 1e-6))
        nearest = int(np.argmin(distance))
        affinity = (self.labels == self.labels[nearest]).astype(np.float32)
        return similarity, affinity


def load_bundle(config: dict) -> dict:
    artifacts = project_path(config, config["artifacts_dir"])
    checkpoint = artifacts / "actor.pt"
    catalog_checkpoint = artifacts / "catalog_actor.pt"
    cache_key = "|".join([
        str(artifacts.resolve()),
        str(checkpoint.stat().st_mtime_ns if checkpoint.exists() else 0),
        str(catalog_checkpoint.stat().st_mtime_ns if catalog_checkpoint.exists() else 0),
    ])
    if cache_key in _BUNDLE_CACHE:
        return _BUNDLE_CACHE[cache_key]
    courses = pd.read_csv(artifacts / "courses.csv", dtype={"course_id": str})
    train = pd.read_csv(artifacts / "train.csv", dtype={"user_id": str, "course_id": str})
    embeddings = np.load(artifacts / "course_embeddings.npy").astype(np.float32)
    semantic_path = artifacts / "course_semantic_embeddings.npy"
    semantic_embeddings = (
        np.load(semantic_path).astype(np.float32)
        if semantic_path.exists()
        else embeddings
    )
    user_features = np.load(artifacts / "user_features.npy").astype(np.float32)
    course_ids = [str(value) for value in load_json(artifacts / "course_ids.json")]
    user_ids = [str(value) for value in load_json(artifacts / "user_ids.json")]
    profiles = learner_profiles(train, user_ids, course_ids, semantic_embeddings)
    course_context = build_course_context(train, course_ids)
    actor = Actor(dropout=config["model"].get("dropout", 0.1))
    if checkpoint.exists():
        actor.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    actor.eval()
    catalog_actor = None
    if catalog_checkpoint.exists():
        catalog_actor = CatalogActor(
            len(course_ids),
            dropout=config["model"].get("dropout", 0.1),
        )
        catalog_actor.load_state_dict(
            torch.load(catalog_checkpoint, map_location="cpu", weights_only=True),
        )
        catalog_actor.eval()
    bundle = {
        "artifacts": artifacts,
        "courses": courses,
        "train": train,
        "embeddings": embeddings,
        "semantic_embeddings": semantic_embeddings,
        "user_features": user_features,
        "course_ids": course_ids,
        "user_ids": user_ids,
        "profiles": profiles,
        "course_context": course_context,
        "actor": actor,
        "catalog_actor": catalog_actor,
    }
    _BUNDLE_CACHE.clear()
    _BUNDLE_CACHE[cache_key] = bundle
    return bundle


def recommend(
    config: dict,
    user_id: str,
    top_k: int | None = None,
    translate_output: bool = True,
) -> pd.DataFrame:
    bundle = load_bundle(config)
    if user_id not in bundle["user_ids"]:
        raise ValueError(f"Unknown user_id: {user_id}")
    top_k = top_k or config["recommendation"].get("top_k", 10)
    user_index = bundle["user_ids"].index(user_id)
    with torch.no_grad():
        use_ppo = config["recommendation"].get("use_ppo", True)
        disable_bert = config["recommendation"].get("disable_bert", False)
        if bundle["catalog_actor"] is not None and use_ppo:
            profile_state = (
                np.zeros_like(bundle["profiles"][user_index])
                if disable_bert
                else bundle["profiles"][user_index]
            )
            policy_state = torch.tensor(np.concatenate([
                bundle["user_features"][user_index],
                profile_state,
            ]), dtype=torch.float32).unsqueeze(0)
            actor_scores = bundle["catalog_actor"](policy_state).squeeze(0).numpy()
        else:
            states = make_states(
                torch.tensor(bundle["user_features"][user_index]),
                torch.tensor(bundle["embeddings"]),
            )
            actor_scores = bundle["actor"](states).numpy()
    semantic = (
        np.zeros(len(bundle["course_ids"]), dtype=np.float32)
        if config["recommendation"].get("disable_bert", False)
        else bundle["semantic_embeddings"] @ bundle["profiles"][user_index]
    )
    cache_key = str(bundle["artifacts"].resolve())
    similarity_model = _SIMILARITY_CACHE.get(cache_key)
    if similarity_model is None:
        similarity_model = SimilarityModel(
            bundle["semantic_embeddings"],
            config["mahalanobis"],
            bundle["course_context"],
        )
        _SIMILARITY_CACHE[cache_key] = similarity_model
    if config["recommendation"].get("disable_bert", False):
        mahalanobis = np.zeros(len(bundle["course_ids"]), dtype=np.float32)
        cluster = np.zeros(len(bundle["course_ids"]), dtype=np.float32)
    else:
        mahalanobis, cluster = similarity_model.scores(
            bundle["profiles"][user_index],
            bundle["user_features"][user_index],
        )
    rec_cfg = config["recommendation"]
    final = (
        actor_scores
        + float(rec_cfg.get("semantic_weight", 0.5)) * semantic
        + float(config["mahalanobis"].get("logit_fusion_alpha", 0.2)) * mahalanobis
        + float(config["mahalanobis"].get("logit_fusion_beta", 0.1)) * cluster
    )
    completed = set(bundle["train"].loc[bundle["train"]["user_id"] == user_id, "course_id"].astype(str))
    for index, course_id in enumerate(bundle["course_ids"]):
        if course_id in completed:
            final[index] = -np.inf
    if rec_cfg.get("action_space", "full_catalog") == "candidate_set":
        size = min(int(rec_cfg.get("candidate_size", 100)), len(bundle["course_ids"]))
        popularity = (
            bundle["train"].groupby("course_id")["reward"].sum()
            .reindex(bundle["course_ids"]).fillna(0.0).to_numpy()
        )
        semantic_count = max(size // 2, 1)
        popular_count = max(size // 4, 1)
        exploration_count = max(size - semantic_count - popular_count, 0)
        candidates = set(np.argsort(-semantic)[:semantic_count])
        candidates.update(np.argsort(-popularity)[:popular_count])
        rng = np.random.default_rng(abs(hash(user_id)) % (2**32))
        available = np.flatnonzero(np.isfinite(final))
        if exploration_count and len(available):
            candidates.update(rng.choice(
                available,
                size=min(exploration_count, len(available)),
                replace=False,
            ).tolist())
        candidate_mask = np.zeros(len(final), dtype=bool)
        candidate_mask[list(candidates)] = True
        final[~candidate_mask] = -np.inf
    top_indices = np.argsort(-final)[:top_k]
    indexed = bundle["courses"].set_index("course_id")
    translator = TopKTranslator(
        config["translation"].get("backend", "offline_glossary"),
        str(project_path(
            config,
            config["translation"].get("local_model_dir", "models/opus-mt-zh-en"),
        )),
    ) if translate_output else None
    rows = []
    for rank, index in enumerate(top_indices, 1):
        course_id = bundle["course_ids"][index]
        course = indexed.loc[course_id]
        rows.append({
            "rank": rank,
            "course_id": course_id,
            "title_en": translator.translate(course.get("title", ""), 220) if translator else "",
            "description_en": translator.translate(course.get("description", ""), 500) if translator else "",
            "original_title_zh": course.get("title", ""),
            "actor_score": float(actor_scores[index]),
            "semantic_similarity": float(semantic[index]),
            "mahalanobis_similarity": float(mahalanobis[index]),
            "cluster_affinity": float(cluster[index]),
            "final_score": float(final[index]),
        })
    output = pd.DataFrame(rows)
    exports = ensure_dir(bundle["artifacts"] / "exports")
    if translate_output:
        output.to_csv(exports / f"recommendations_{user_id}.csv", index=False, encoding="utf-8-sig")
    return output
