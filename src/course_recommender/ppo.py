from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as functional
from torch.distributions import Categorical

from .models import CatalogActor, Critic


@dataclass
class RolloutBuffer:
    states: list[np.ndarray] = field(default_factory=list)
    masks: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)

    def clear(self) -> None:
        for value in vars(self).values():
            value.clear()

    def advantages(
        self,
        next_value: float,
        gamma: float,
        gae_lambda: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        advantages = np.zeros(len(self.rewards), dtype=np.float32)
        gae = 0.0
        for index in reversed(range(len(self.rewards))):
            following = next_value if index == len(self.rewards) - 1 else self.values[index + 1]
            non_terminal = 1.0 - float(self.dones[index])
            delta = self.rewards[index] + gamma * following * non_terminal - self.values[index]
            gae = delta + gamma * gae_lambda * non_terminal * gae
            advantages[index] = gae
        returns = advantages + np.asarray(self.values, dtype=np.float32)
        return torch.tensor(advantages), torch.tensor(returns)


def collect_rollout(env, actor: CatalogActor, critic: Critic, steps: int, device: torch.device) -> RolloutBuffer:
    buffer = RolloutBuffer()
    observation, _ = env.reset()
    for _ in range(steps):
        state = torch.tensor(observation["state"], dtype=torch.float32, device=device).unsqueeze(0)
        mask = torch.tensor(observation["action_mask"], dtype=torch.bool, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = actor(state, mask)
            distribution = Categorical(logits=logits)
            action = distribution.sample()
            value = critic(state)
        next_observation, reward, terminated, truncated, _ = env.step(int(action.item()))
        buffer.states.append(observation["state"])
        buffer.masks.append(observation["action_mask"])
        buffer.actions.append(int(action.item()))
        buffer.log_probs.append(float(distribution.log_prob(action).item()))
        buffer.rewards.append(float(reward))
        buffer.values.append(float(value.item()))
        buffer.dones.append(bool(terminated or truncated))
        observation = next_observation
        if terminated or truncated:
            observation, _ = env.reset()
    return buffer


def ppo_update(
    actor: CatalogActor,
    critic: Critic,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    device: torch.device,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_epsilon: float = 0.2,
    update_epochs: int = 4,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    normalize_advantages: bool = True,
) -> dict[str, float]:
    advantages, returns = buffer.advantages(0.0, gamma, gae_lambda)
    if normalize_advantages:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    states = torch.tensor(np.asarray(buffer.states), dtype=torch.float32, device=device)
    masks = torch.tensor(np.asarray(buffer.masks), dtype=torch.bool, device=device)
    actions = torch.tensor(buffer.actions, dtype=torch.long, device=device)
    old_log_probs = torch.tensor(buffer.log_probs, dtype=torch.float32, device=device)
    advantages, returns = advantages.to(device), returns.to(device)
    totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "kl": 0.0}
    for _ in range(update_epochs):
        distribution = Categorical(logits=actor(states, masks))
        log_probs = distribution.log_prob(actions)
        ratio = torch.exp(log_probs - old_log_probs)
        clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
        policy_loss = -torch.minimum(ratio * advantages, clipped * advantages).mean()
        values = critic(states)
        value_loss = functional.mse_loss(values, returns)
        entropy = distribution.entropy().mean()
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 1.0)
        optimizer.step()
        totals["policy_loss"] += float(policy_loss.detach())
        totals["value_loss"] += float(value_loss.detach())
        totals["entropy"] += float(entropy.detach())
        totals["kl"] += float((old_log_probs - log_probs).mean().detach())
    return {key: value / update_epochs for key, value in totals.items()}
