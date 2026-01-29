from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class DynVAE_CatA(nn.Module):
    def __init__(
        self,
        dim_x: int = 20,
        dim_z: int = 3,
        num_actions: int = 6,
        a_emb_dim: int = 8,
        transition_dim: int = 64,
        hidden_y: int = 64,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dim_x = int(dim_x)
        self.dim_z = int(dim_z)
        self.K = int(num_actions)
        self.a_emb_dim = int(a_emb_dim)
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.a_emb = nn.Embedding(self.K, self.a_emb_dim)
        self.z1_loc = nn.Parameter(torch.zeros(self.dim_z))
        self.z1_unconstrained_scale = nn.Parameter(torch.zeros(self.dim_z))
        self.softplus = nn.Softplus()

        self.p_z_net = nn.Sequential(
            nn.Linear(self.dim_z + self.a_emb_dim, transition_dim),
            nn.Tanh(),
            nn.Linear(transition_dim, self.dim_z * 2),
        )
        self.p_y = nn.Sequential(
            nn.Linear(self.dim_z + self.K, hidden_y),
            nn.Tanh(),
            nn.Linear(hidden_y, 1),
        )

        self.to(self.device)

    def p_z(self, z: torch.Tensor, a_emb: torch.Tensor):
        inputs = torch.cat([z, a_emb], dim=-1)
        out = self.p_z_net(inputs)
        loc, scale_logits = torch.chunk(out, 2, dim=-1)
        scale = self.softplus(scale_logits) + 1e-4
        return loc, scale

    def p_z1_params(self, batch_size: int = 1, device: Optional[torch.device] = None):
        device = device or self.device
        loc = self.z1_loc.unsqueeze(0).expand(batch_size, -1).to(device)
        scale = (
            self.softplus(self.z1_unconstrained_scale)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .to(device)
            + 1e-4
        )
        return loc, scale

    @torch.no_grad()
    def sample_z1(self, batch_size: int = 1):
        loc, scale = self.p_z1_params(batch_size, self.device)
        z = torch.distributions.Normal(loc, scale).sample()
        return z.squeeze(0) if batch_size == 1 else z
