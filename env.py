from __future__ import annotations

from typing import Callable, Optional, Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym


RewardFn = Callable[
    [torch.Tensor, int, int, int, Any, float, Optional[Sequence[float]]],
    float,
]


def base_reward_fn(
    z_t: torch.Tensor,
    action: int,
    current_t: int,
    max_T: int,
    iter_model,
    beta: float,
    dose_values: Optional[Sequence[float]],
) -> float:
    if dose_values is None:
        cost = float(action)
    else:
        cost = float(dose_values[action])

    if current_t < max_T:
        return -beta * cost

    device = iter_model.device
    z_t = z_t.to(device).unsqueeze(0)
    a_onehot = F.one_hot(
        torch.tensor([action], dtype=torch.long, device=device),
        num_classes=iter_model.K,
    ).float()
    logits = iter_model.p_y(torch.cat([z_t, a_onehot], dim=-1))
    risk = torch.sigmoid(logits).squeeze(0).squeeze(0)
    return float(-risk.item() - beta * cost)


class TransfusionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        iter_model,
        max_T: int = 5,
        reward_fn: RewardFn = base_reward_fn,
        beta: float = 0.0,
        dose_values: Optional[Sequence[float]] = None,
    ):
        self.iter_model = iter_model
        self.max_T = int(max_T)
        self.reward_fn = reward_fn
        self.beta = float(beta)
        self.dose_values = dose_values

        self.current_z = None
        self.current_t = 1

        self.K = int(iter_model.K)
        self.dim_z = int(iter_model.dim_z)
        self.action_space = gym.spaces.Discrete(self.K)
        self.observation_space = gym.spaces.Box(
            low=-1e9, high=1e9, shape=(self.dim_z,), dtype=np.float32
        )

    def _obs(self) -> np.ndarray:
        if torch.is_tensor(self.current_z):
            return self.current_z.detach().cpu().numpy().astype(np.float32)
        return np.asarray(self.current_z, dtype=np.float32)

    @torch.no_grad()
    def _sample_next_z(self, z: torch.Tensor, a: int) -> torch.Tensor:
        device = self.iter_model.device
        z = z.to(device).unsqueeze(0)
        a_idx = torch.tensor([a], dtype=torch.long, device=device)
        a_emb = self.iter_model.a_emb(a_idx)
        loc, scale = self.iter_model.p_z(z, a_emb)
        scale = torch.clamp(scale, min=1e-6)
        return torch.distributions.Normal(loc, scale).sample().squeeze(0)

    def reset(
        self,
        *,
        start_t: int,
        z_t: torch.Tensor,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.current_z = z_t
        self.current_t = int(start_t)
        return self._obs(), {"t": self.current_t}

    def step(self, action: int):
        if self.current_z is None:
            raise RuntimeError("Call reset() before step().")

        z_t = self.current_z
        z_next = self._sample_next_z(z_t, int(action))
        self.current_z = z_next

        terminated = self.current_t >= self.max_T
        reward = self.reward_fn(
            z_t,
            int(action),
            self.current_t,
            self.max_T,
            self.iter_model,
            self.beta,
            self.dose_values,
        )
        info = {"t": self.current_t}
        if terminated:
            device = self.iter_model.device
            z_term = z_t.to(device).unsqueeze(0)
            a_onehot = F.one_hot(
                torch.tensor([action], dtype=torch.long, device=device),
                num_classes=self.iter_model.K,
            ).float()
            logits = self.iter_model.p_y(torch.cat([z_term, a_onehot], dim=-1))
            aki_prob = torch.sigmoid(logits).squeeze(0).squeeze(0)
            info["aki_prob"] = float(aki_prob.item())

        if not terminated:
            self.current_t += 1

        return self._obs(), float(reward), terminated, False, info
