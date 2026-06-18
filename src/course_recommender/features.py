from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import project_path
from .utils import dump_json, ensure_dir


USER_FEATURES = [
    "average_completion_rate",
    "average_quiz_score",
    "normalized_engagement_time",
    "average_rating",
    "normalized_click_count",
    "normalized_video_view_count",
    "normalized_access_frequency",
    "normalized_session_duration",
    "difficulty_preference",
    "recent_activity_score",
]


def hash_embeddings(texts: list[str], dim: int = 768) -> np.ndarray:
    matrix = np.zeros((len(texts), dim), dtype=np.float32)
    for row, text in enumerate(texts):
        for token in str(text).lower().split()[:256]:
            digest = hashlib.sha256(token.encode("utf-8", errors="ignore")).digest()
            column = int.from_bytes(digest[:4], "little") % dim
            matrix[row, column] += 1.0 if digest[4] % 2 == 0 else -1.0
        norm = np.linalg.norm(matrix[row])
        if norm:
            matrix[row] /= norm
    return matrix


def bert_embeddings(texts: list[str], model_name: str, batch_size: int, max_length: int) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, local_files_only=True, use_fast=False,
    )
    model = AutoModel.from_pretrained(model_name, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    batches = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start:start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            batches.append(model(**encoded).last_hidden_state[:, 0].cpu().numpy())
    embeddings = np.vstack(batches).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-8)


def build_user_features(train: pd.DataFrame, user_ids: list[str]) -> tuple[np.ndarray, dict]:
    rows = []
    train = train.copy()
    train["timestamp"] = pd.to_datetime(train["timestamp"], errors="coerce", utc=True)
    newest = train["timestamp"].max()
    grouped = {key: value for key, value in train.groupby("user_id")}
    for user_id in user_ids:
        group = grouped.get(user_id)
        if group is None or group.empty:
            rows.append([0.0] * 10)
            continue
        rating = pd.to_numeric(group["rating"], errors="coerce").mean()
        last = group["timestamp"].max()
        recency_days = (newest - last).days if pd.notna(newest) and pd.notna(last) else 365
        rows.append([
            group["completion_rate"].mean(),
            group["quiz_score"].mean(),
            group["engagement_time"].mean(),
            0.5 if pd.isna(rating) else min(max(rating / 5.0, 0.0), 1.0),
            group["clicked"].sum(),
            group["video_views"].sum(),
            group["access_frequency"].mean(),
            group["session_duration"].mean(),
            0.5,
            1.0 / (1.0 + max(recency_days, 0)),
        ])
    raw = np.nan_to_num(np.asarray(rows, dtype=np.float32), nan=0.0)
    mins, maxs = raw.min(axis=0), raw.max(axis=0)
    ranges = np.where(maxs - mins > 1e-8, maxs - mins, 1.0)
    scaled = np.clip((raw - mins) / ranges, 0.0, 1.0)
    return scaled, {"columns": USER_FEATURES, "min": mins.tolist(), "max": maxs.tolist()}


def build_features(config: dict) -> dict:
    artifacts = ensure_dir(project_path(config, config["artifacts_dir"]))
    courses = pd.read_csv(artifacts / "courses.csv", dtype={"course_id": str})
    users = pd.read_csv(artifacts / "users.csv", dtype={"user_id": str})
    train = pd.read_csv(artifacts / "train.csv", dtype={"user_id": str, "course_id": str})
    text = courses["text"].fillna("").astype(str).tolist()
    model_cfg = config["embedding"]
    backend = model_cfg.get("backend", "hash")
    cache_meta_path = artifacts / "embedding_cache_meta.json"
    text_hash = hashlib.sha256("\n".join(text).encode("utf-8", errors="ignore")).hexdigest()
    if backend == "bert":
        checkpoint = project_path(config, model_cfg.get("checkpoint_dir", ""))
        local_model = project_path(config, model_cfg["local_model_dir"])
        model_path = checkpoint if (checkpoint / "config.json").exists() else local_model
        model_marker = model_path / "model.safetensors"
        cache_key = hashlib.sha256(json.dumps({
            "embedding_algorithm": "cls_l2_normalized_v2",
            "model": str(model_path),
            "model_size": model_marker.stat().st_size if model_marker.exists() else 0,
            "text_hash": text_hash,
            "max_length": model_cfg.get("max_length", 128),
        }, sort_keys=True).encode()).hexdigest()
        embedding_path = artifacts / "course_embeddings.npy"
        if embedding_path.exists() and cache_meta_path.exists():
            cache_meta = json.loads(cache_meta_path.read_text(encoding="utf-8"))
            if cache_meta.get("cache_key") == cache_key:
                embeddings = np.load(embedding_path).astype(np.float32)
            else:
                embeddings = bert_embeddings(
                    text,
                    str(model_path),
                    model_cfg.get("batch_size", 16),
                    model_cfg.get("max_length", 128),
                )
        else:
            embeddings = bert_embeddings(
                text,
                str(model_path),
                model_cfg.get("batch_size", 16),
                model_cfg.get("max_length", 128),
            )
        dump_json(cache_meta_path, {
            "cache_key": cache_key,
            "model_path": str(model_path),
            "dataset_text_sha256": text_hash,
            "max_length": model_cfg.get("max_length", 128),
        })
    else:
        embeddings = hash_embeddings(text, 768)
    user_ids = users["user_id"].astype(str).tolist()
    user_features, feature_meta = build_user_features(train, user_ids)
    np.save(artifacts / "course_embeddings.npy", embeddings)
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    centered /= np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), 1e-8)
    np.save(artifacts / "course_semantic_embeddings.npy", centered.astype(np.float32))
    np.save(artifacts / "user_features.npy", user_features)
    dump_json(artifacts / "course_ids.json", courses["course_id"].astype(str).tolist())
    dump_json(artifacts / "user_ids.json", user_ids)
    dump_json(artifacts / "feature_meta.json", feature_meta)
    result = {
        "embedding_backend": backend,
        "course_embedding_shape": list(embeddings.shape),
        "user_feature_shape": list(user_features.shape),
        "state_dimension": int(embeddings.shape[1] + user_features.shape[1]),
    }
    dump_json(artifacts / "feature_summary.json", result)
    return result
