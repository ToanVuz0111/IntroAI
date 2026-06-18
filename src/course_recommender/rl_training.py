from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, TensorDataset

from .config import project_path
from .environment import CourseRecommendationEnv
from .models import CatalogActor, Critic
from .ppo import collect_rollout, ppo_update
from .utils import dump_json, load_json, set_seed


def _supervised_transitions(
    train: pd.DataFrame,
    features: np.ndarray,
    user_ids: list[str],
    course_ids: list[str],
    embeddings: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    user_index = {value: index for index, value in enumerate(user_ids)}
    course_index = {value: index for index, value in enumerate(course_ids)}
    states, targets = [], []
    for user_id, group in train.sort_values("timestamp").groupby("user_id"):
        ordered = [str(value) for value in group["course_id"] if str(value) in course_index]
        if len(ordered) < 2 or str(user_id) not in user_index:
            continue
        history = []
        for target in ordered:
            if history:
                profile = embeddings[[course_index[value] for value in history]].mean(axis=0)
                profile /= max(np.linalg.norm(profile), 1e-8)
                states.append(np.concatenate([features[user_index[str(user_id)]], profile]))
                targets.append(course_index[target])
            history.append(target)
    return torch.tensor(np.asarray(states), dtype=torch.float32), torch.tensor(targets, dtype=torch.long)


def train_ppo_environment(config: dict) -> dict:
    set_seed(config.get("seed", 42))
    artifacts = project_path(config, config["artifacts_dir"])
    train = pd.read_csv(artifacts / "train.csv", dtype={"user_id": str, "course_id": str})
    validation = pd.read_csv(artifacts / "val.csv", dtype={"user_id": str, "course_id": str})
    semantic_path = artifacts / "course_semantic_embeddings.npy"
    embeddings = np.load(
        semantic_path if semantic_path.exists() else artifacts / "course_embeddings.npy",
    ).astype(np.float32)
    features = np.load(artifacts / "user_features.npy").astype(np.float32)
    user_ids = [str(value) for value in load_json(artifacts / "user_ids.json")]
    course_ids = [str(value) for value in load_json(artifacts / "course_ids.json")]
    cfg = config.get("environment", {})
    env = CourseRecommendationEnv(
        train,
        features,
        user_ids,
        course_ids,
        embeddings,
        max_steps=int(cfg.get("max_steps", 10)),
        response_mode=cfg.get("response_mode", "synthetic_simulator"),
        seed=config.get("seed", 42),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = CatalogActor(len(course_ids), dropout=config["model"].get("dropout", 0.1)).to(device)
    critic = Critic().to(device)
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=float(config["training"].get("learning_rate", 1e-4)),
    )
    supervised_states, supervised_targets = _supervised_transitions(
        train, features, user_ids, course_ids, embeddings,
    )
    validation_states, validation_targets = _supervised_transitions(
        validation, features, user_ids, course_ids, embeddings,
    )
    pretrain_epochs = int(cfg.get("supervised_pretrain_epochs", 10))
    iterations = int(cfg.get("ppo_iterations", 20))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(pretrain_epochs + iterations, 1),
        eta_min=float(cfg.get("minimum_learning_rate", 1e-6)),
    )
    start_time = time.perf_counter()
    if len(supervised_states):
        loader = DataLoader(
            TensorDataset(supervised_states, supervised_targets),
            batch_size=int(config["training"].get("batch_size", 128)),
            shuffle=True,
        )
        actor.train()
        for _ in range(pretrain_epochs):
            for states, targets in loader:
                states, targets = states.to(device), targets.to(device)
                loss = functional.cross_entropy(actor(states), targets)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
    history = []
    patience = int(cfg.get("early_stopping_patience", 10))
    best_validation = float("inf")
    best_actor = copy.deepcopy(actor.state_dict())
    best_critic = copy.deepcopy(critic.state_dict())
    stale_iterations = 0
    for iteration in range(iterations):
        rollout = collect_rollout(env, actor, critic, int(cfg.get("rollout_steps", 512)), device)
        if cfg.get("normalize_rewards", True) and len(rollout.rewards) > 1:
            rewards = np.asarray(rollout.rewards, dtype=np.float32)
            normalized = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            rollout.rewards = normalized.tolist()
        metrics = ppo_update(
            actor,
            critic,
            optimizer,
            rollout,
            device,
            gamma=float(config["training"].get("discount_factor", 0.99)),
            gae_lambda=float(cfg.get("gae_lambda", 0.95)),
            clip_epsilon=float(config["training"].get("ppo_clip_epsilon", 0.2)),
            update_epochs=int(cfg.get("update_epochs", 4)),
            entropy_coef=float(cfg.get("entropy_coefficient", 0.01)),
            normalize_advantages=bool(cfg.get("normalize_advantages", True)),
        )
        scheduler.step()
        actor.eval()
        with torch.no_grad():
            if len(validation_states):
                validation_loss = float(functional.cross_entropy(
                    actor(validation_states.to(device)),
                    validation_targets.to(device),
                ))
            elif len(supervised_states):
                validation_loss = float(functional.cross_entropy(
                    actor(supervised_states[: min(len(supervised_states), 1024)].to(device)),
                    supervised_targets[: min(len(supervised_targets), 1024)].to(device),
                ))
            else:
                validation_loss = float(-np.mean(rollout.rewards))
        actor.train()
        history.append({
            "iteration": iteration + 1,
            "mean_reward": float(np.mean(rollout.rewards)),
            "validation_loss": validation_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **metrics,
        })
        if validation_loss < best_validation - float(cfg.get("early_stopping_min_delta", 1e-4)):
            best_validation = validation_loss
            best_actor = copy.deepcopy(actor.state_dict())
            best_critic = copy.deepcopy(critic.state_dict())
            stale_iterations = 0
        else:
            stale_iterations += 1
            if stale_iterations >= patience:
                break
    actor.load_state_dict(best_actor)
    critic.load_state_dict(best_critic)
    torch.save(actor.state_dict(), artifacts / "catalog_actor.pt")
    torch.save(critic.state_dict(), artifacts / "ppo_critic.pt")
    pd.DataFrame(history).to_csv(artifacts / "ppo_training_history.csv", index=False)
    summary = {
        "iterations": iterations,
        "iterations_completed": len(history),
        "catalog_size": len(course_ids),
        "device": str(device),
        "response_mode": env.response_mode,
        "supervised_transitions": len(supervised_states),
        "supervised_pretrain_epochs": pretrain_epochs,
        "final_mean_reward": history[-1]["mean_reward"],
        "best_validation_loss": best_validation,
        "early_stopping_patience": patience,
        "cosine_scheduler": True,
        "training_seconds": time.perf_counter() - start_time,
    }
    dump_json(artifacts / "ppo_training_summary.json", summary)
    return summary
