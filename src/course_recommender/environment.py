from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


def weighted_profile(
    history: pd.DataFrame,
    course_index: dict[str, int],
    embeddings: np.ndarray,
) -> np.ndarray:
    vectors, weights = [], []
    for row in history.itertuples(index=False):
        index = course_index.get(str(row.course_id))
        if index is None:
            continue
        vectors.append(embeddings[index])
        reward = float(getattr(row, "reward", 0.0))
        weights.append(max(reward, 1e-3))
    if not vectors:
        return np.zeros(embeddings.shape[1], dtype=np.float32)
    profile = np.average(np.vstack(vectors), axis=0, weights=np.asarray(weights))
    return (profile / max(np.linalg.norm(profile), 1e-8)).astype(np.float32)


class CourseRecommendationEnv(gym.Env):
    """Gymnasium environment using train-only histories and explicit simulation."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        interactions: pd.DataFrame,
        user_features: np.ndarray,
        user_ids: list[str],
        course_ids: list[str],
        course_embeddings: np.ndarray,
        max_steps: int = 10,
        response_mode: str = "synthetic_simulator",
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.interactions = interactions.copy()
        self.interactions["timestamp"] = pd.to_datetime(
            self.interactions["timestamp"], errors="coerce", utc=True,
        )
        self.user_features = user_features.astype(np.float32)
        self.user_ids = user_ids
        self.user_index = {value: index for index, value in enumerate(user_ids)}
        self.course_ids = course_ids
        self.course_index = {value: index for index, value in enumerate(course_ids)}
        self.embeddings = course_embeddings.astype(np.float32)
        self.max_steps = max_steps
        self.response_mode = response_mode
        self.rng = np.random.default_rng(seed)
        self.action_space = spaces.Discrete(len(course_ids))
        self.observation_space = spaces.Dict({
            "state": spaces.Box(-1.0, 1.0, shape=(778,), dtype=np.float32),
            "action_mask": spaces.MultiBinary(len(course_ids)),
        })
        self._groups = {
            str(user_id): group.sort_values("timestamp").reset_index(drop=True)
            for user_id, group in self.interactions.groupby("user_id")
            if str(user_id) in self.user_index
        }
        self._eligible_users = [value for value in user_ids if value in self._groups]
        self.user_id = ""
        self.history = self.interactions.iloc[:0]
        self.target_history = self.interactions.iloc[:0]
        self.profile = np.zeros(768, dtype=np.float32)
        self.mask = np.ones(len(course_ids), dtype=np.int8)
        self.step_count = 0

    def _observation(self) -> dict[str, np.ndarray]:
        user = self.user_features[self.user_index[self.user_id]]
        state = np.concatenate([user, self.profile]).astype(np.float32)
        return {"state": state, "action_mask": self.mask.copy()}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        requested = (options or {}).get("user_id")
        self.user_id = requested if requested in self._groups else str(self.rng.choice(self._eligible_users))
        full = self._groups[self.user_id]
        split = max(1, int(len(full) * 0.7))
        self.history = full.iloc[:split].copy()
        self.target_history = full.iloc[split:].copy()
        self.profile = weighted_profile(self.history, self.course_index, self.embeddings)
        self.mask = np.ones(len(self.course_ids), dtype=np.int8)
        for course_id in self.history["course_id"].astype(str):
            index = self.course_index.get(course_id)
            if index is not None:
                self.mask[index] = 0
        self.step_count = 0
        return self._observation(), {"user_id": self.user_id}

    def _response(self, action: int) -> tuple[float, dict]:
        course_id = self.course_ids[action]
        logged = self.target_history[self.target_history["course_id"].astype(str) == course_id]
        if not logged.empty:
            row = logged.iloc[0]
            return float(row["reward"]), {
                "source": "logged",
                "completion": float(row["completion_rate"]),
                "quiz": float(row["quiz_score"]),
                "engagement": float(row["engagement_time"]),
            }
        if self.response_mode == "logged_replay":
            return 0.0, {"source": "unobserved"}
        similarity = float(np.clip(self.embeddings[action] @ self.profile, -1.0, 1.0))
        preference = (similarity + 1.0) / 2.0
        completion = float(np.clip(self.rng.normal(preference, 0.10), 0.0, 1.0))
        quiz = float(np.clip(self.rng.normal(0.8 * preference + 0.1, 0.12), 0.0, 1.0))
        engagement = float(np.clip(self.rng.normal(preference, 0.15), 0.0, 1.0))
        return 0.4 * completion + 0.4 * quiz + 0.2 * engagement, {
            "source": "synthetic_simulator",
            "completion": completion,
            "quiz": quiz,
            "engagement": engagement,
        }

    def step(self, action: int):
        if not self.action_space.contains(action) or not self.mask[action]:
            return self._observation(), -1.0, False, True, {"invalid_action": True}
        reward, response = self._response(action)
        self.mask[action] = 0
        blend = min(max(reward, 0.05), 1.0)
        profile = (1.0 - blend) * self.profile + blend * self.embeddings[action]
        self.profile = (profile / max(np.linalg.norm(profile), 1e-8)).astype(np.float32)
        self.step_count += 1
        terminated = self.step_count >= self.max_steps or not self.mask.any()
        return self._observation(), float(reward), terminated, False, {
            "course_id": self.course_ids[action],
            **response,
        }
