from __future__ import annotations

import torch
from torch import nn


class Actor(nn.Module):
    def __init__(self, input_dim: int = 778, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(states).squeeze(-1)


class Critic(nn.Module):
    def __init__(self, input_dim: int = 778) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(states).squeeze(-1)


class CatalogActor(nn.Module):
    """Paper-style policy: one probability/logit per course in the catalog."""

    def __init__(self, num_courses: int, input_dim: int = 778, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_courses),
        )

    def forward(
        self,
        states: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = self.network(states)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.bool(), torch.finfo(logits.dtype).min)
        return logits


def make_states(user_features: torch.Tensor, course_embeddings: torch.Tensor) -> torch.Tensor:
    if user_features.ndim == 1:
        user_features = user_features.unsqueeze(0).expand(course_embeddings.shape[0], -1)
    states = torch.cat([user_features, course_embeddings], dim=1)
    if states.shape[1] != 778:
        raise ValueError(f"Expected 778-dimensional state, received {states.shape[1]}")
    return states
