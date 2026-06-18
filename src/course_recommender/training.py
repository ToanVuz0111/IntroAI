from __future__ import annotations

import copy
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, TensorDataset

from .config import project_path
from .models import Actor, Critic
from .utils import dump_json, load_json, set_seed


def _build_rank_dataset(
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    features: np.ndarray,
    users: dict[str, int],
    courses: dict[str, int],
    negatives_per_positive: int,
    seed: int,
    max_samples: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = random.Random(seed)
    all_courses = list(courses)
    seen = {
        str(user_id): set(group["course_id"].astype(str))
        for user_id, group in frame.groupby("user_id")
    }
    rows = frame
    if max_samples:
        rows = rows.sample(min(max_samples, len(rows)), random_state=seed)
    states, labels, returns = [], [], []
    for row in rows.itertuples(index=False):
        user_id, course_id = str(row.user_id), str(row.course_id)
        user_index, course_index = users.get(user_id), courses.get(course_id)
        if user_index is None or course_index is None:
            continue
        states.append(np.concatenate([features[user_index], embeddings[course_index]]))
        labels.append(1.0)
        returns.append(float(row.reward))
        candidates = [value for value in all_courses if value not in seen.get(user_id, set())]
        for _ in range(negatives_per_positive):
            if not candidates:
                break
            negative_id = rng.choice(candidates)
            states.append(np.concatenate([features[user_index], embeddings[courses[negative_id]]]))
            labels.append(0.0)
            returns.append(0.0)
    return (
        torch.tensor(np.asarray(states), dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32),
        torch.tensor(returns, dtype=torch.float32),
    )


def _validation_loss(actor: Actor, states: torch.Tensor, labels: torch.Tensor) -> float:
    actor.eval()
    with torch.no_grad():
        result = float(functional.binary_cross_entropy_with_logits(actor(states), labels))
    actor.train()
    return result


def train(config: dict) -> dict:
    seed = config.get("seed", 42)
    set_seed(seed)
    artifacts = project_path(config, config["artifacts_dir"])
    embeddings = np.load(artifacts / "course_embeddings.npy").astype(np.float32)
    features = np.load(artifacts / "user_features.npy").astype(np.float32)
    train_df = pd.read_csv(artifacts / "train.csv", dtype={"user_id": str, "course_id": str})
    val_df = pd.read_csv(artifacts / "val.csv", dtype={"user_id": str, "course_id": str})
    users = {str(value): index for index, value in enumerate(load_json(artifacts / "user_ids.json"))}
    courses = {str(value): index for index, value in enumerate(load_json(artifacts / "course_ids.json"))}
    train_cfg = config["training"]
    negatives = int(train_cfg.get("negative_samples_per_positive", 1))
    train_states, train_labels, train_returns = _build_rank_dataset(
        train_df, embeddings, features, users, courses, negatives, seed, train_cfg.get("max_samples"),
    )
    val_states, val_labels, _ = _build_rank_dataset(
        val_df, embeddings, features, users, courses, negatives, seed + 1,
    )
    if train_states.shape[1] != 778:
        raise RuntimeError(f"Expected state dimension 778, received {train_states.shape[1]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor, critic = Actor().to(device), Critic().to(device)
    old_actor = copy.deepcopy(actor).eval()
    parameters = list(actor.parameters()) + list(critic.parameters())
    optimizer = torch.optim.Adam(parameters, lr=float(train_cfg.get("learning_rate", 1e-4)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(int(train_cfg.get("epochs", 100)), 1),
    )
    loader = DataLoader(
        TensorDataset(train_states, train_labels, train_returns),
        batch_size=int(train_cfg.get("batch_size", 128)),
        shuffle=True,
    )
    clip_epsilon = float(train_cfg.get("ppo_clip_epsilon", 0.2))
    patience = int(train_cfg.get("early_stopping_patience", 10))
    history = []
    best_validation = float("inf")
    best_actor = None
    best_critic = None
    stale_epochs = 0
    started = time.perf_counter()

    for epoch in range(int(train_cfg.get("epochs", 100))):
        actor.train()
        critic.train()
        totals = {"loss": 0.0, "policy": 0.0, "value": 0.0, "bce": 0.0}
        for states, labels, returns in loader:
            states, labels, returns = states.to(device), labels.to(device), returns.to(device)
            logits = actor(states)
            values = critic(states)
            with torch.no_grad():
                old_logits = old_actor(states)
                old_log_probability = (
                    labels * functional.logsigmoid(old_logits)
                    + (1.0 - labels) * functional.logsigmoid(-old_logits)
                )
            log_probability = (
                labels * functional.logsigmoid(logits)
                + (1.0 - labels) * functional.logsigmoid(-logits)
            )
            advantages = returns - values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            ratio = torch.exp(log_probability - old_log_probability)
            clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
            policy_loss = -torch.minimum(ratio * advantages, clipped_ratio * advantages).mean()
            bce_loss = functional.binary_cross_entropy_with_logits(logits, labels)
            value_loss = functional.mse_loss(values, returns)
            loss = policy_loss + bce_loss + 0.5 * value_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            totals["loss"] += float(loss.detach())
            totals["policy"] += float(policy_loss.detach())
            totals["value"] += float(value_loss.detach())
            totals["bce"] += float(bce_loss.detach())
        scheduler.step()
        old_actor.load_state_dict(actor.state_dict())
        validation = (
            _validation_loss(actor, val_states.to(device), val_labels.to(device))
            if len(val_states) else totals["bce"] / max(len(loader), 1)
        )
        row = {
            "epoch": epoch + 1,
            **{key: value / max(len(loader), 1) for key, value in totals.items()},
            "validation_bce": validation,
        }
        history.append(row)
        if validation < best_validation - 1e-5:
            best_validation = validation
            best_actor = copy.deepcopy(actor.state_dict())
            best_critic = copy.deepcopy(critic.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    actor.load_state_dict(best_actor or actor.state_dict())
    critic.load_state_dict(best_critic or critic.state_dict())
    torch.save(actor.state_dict(), artifacts / "actor.pt")
    torch.save(critic.state_dict(), artifacts / "critic.pt")
    pd.DataFrame(history).to_csv(artifacts / "training_history.csv", index=False)
    result = {
        "samples": len(train_states),
        "positive_samples": int(train_labels.sum()),
        "negative_samples": int((1 - train_labels).sum()),
        "state_dimension": train_states.shape[1],
        "epochs_completed": len(history),
        "best_validation_bce": best_validation,
        "actor_parameters": sum(item.numel() for item in actor.parameters()),
        "critic_parameters": sum(item.numel() for item in critic.parameters()),
        "device": str(device),
        "training_seconds": time.perf_counter() - started,
        "seconds_per_epoch": (time.perf_counter() - started) / max(len(history), 1),
    }
    dump_json(artifacts / "training_summary.json", result)
    return result
