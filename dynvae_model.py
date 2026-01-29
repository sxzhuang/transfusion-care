import argparse
import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam

# Import data loading functions
from load_data import create_loaders


# ============================================================
# Utils
# ============================================================

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def reverse_padded_sequence(seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """
    Reverse each sequence up to its length, keep padding.
    seq: (B, T, D)
    lengths: (B,)
    """
    B, T, D = seq.shape
    out = seq.clone()
    for i in range(B):
        L = int(lengths[i].item())
        if L > 0:
            out[i, :L, :] = torch.flip(seq[i, :L, :], dims=[0])
        if L < T:
            out[i, L:, :] = 0.0
    return out


# ============================================================
# Neural components
# ============================================================

class MLPEmbed(nn.Module):
    """Embed static covariates v -> v_emb."""
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPNormal(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, min_scale=0.1, max_scale=0.8):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.loc = nn.Linear(hidden_dim, out_dim)
        self.scale = nn.Linear(hidden_dim, out_dim)
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)

    def forward(self, x):
        h = self.tanh(self.fc1(x))
        h = self.tanh(self.fc2(h))
        loc = self.loc(h)
        scale = (self.softplus(self.scale(h)) + self.min_scale).clamp(max=self.max_scale)
        return loc, scale


class MLPLogits(nn.Module):
    def __init__(self, in_dim, out_dim=1, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        h = self.tanh(self.fc1(x))
        h = self.tanh(self.fc2(h))
        return self.out(h)


class MLPCat(nn.Module):
    """Categorical logits from an MLP."""
    def __init__(self, in_dim, num_classes, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_classes)
        self.tanh = nn.Tanh()

    def forward(self, x):
        h = self.tanh(self.fc1(x))
        h = self.tanh(self.fc2(h))
        return self.out(h)  # logits


class ContextCombiner(nn.Module):
    """
    q(z_t | context, g_t):
      context = [z_{t-1}, emb(a_{t-1}), v_emb]
      g_t = backward RNN message over [x,a,y,v]
    """
    def __init__(self, context_dim, g_dim, z_dim):
        super().__init__()
        self.lin_context_to_g = nn.Linear(context_dim, g_dim)
        self.lin_g_to_loc = nn.Linear(g_dim, z_dim)
        self.lin_g_to_scale = nn.Linear(g_dim, z_dim)
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, context, g_t):
        h = 0.5 * (self.tanh(self.lin_context_to_g(context)) + g_t)
        loc = self.lin_g_to_loc(h)
        scale = self.softplus(self.lin_g_to_scale(h)) + 1e-4
        return loc, scale


class GatedTransitionZCatAV(nn.Module):
    """
    DMM-style gated transition:
      p(z_t | z_{t-1}, a_{t-1}, v) = Normal(loc, scale)
    """
    def __init__(self, z_dim, a_emb_dim, v_emb_dim, transition_dim=64):
        super().__init__()
        in_dim = z_dim + a_emb_dim + v_emb_dim

        self.lin_gate = nn.Linear(in_dim, transition_dim)
        self.lin_gate_out = nn.Linear(transition_dim, z_dim)

        self.lin_prop = nn.Linear(in_dim, transition_dim)
        self.lin_prop_out = nn.Linear(transition_dim, z_dim)

        self.lin_sig = nn.Linear(z_dim, z_dim)
        self.lin_z_to_loc = nn.Linear(z_dim, z_dim)

        # identity init
        self.lin_z_to_loc.weight.data = torch.eye(z_dim)
        self.lin_z_to_loc.bias.data = torch.zeros(z_dim)

        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, z_tm1, a_tm1_emb, v_emb):
        zav = torch.cat([z_tm1, a_tm1_emb, v_emb], dim=-1)
        _gate = self.relu(self.lin_gate(zav))
        gate = torch.sigmoid(self.lin_gate_out(_gate))

        _prop = self.relu(self.lin_prop(zav))
        prop_mean = self.lin_prop_out(_prop)

        loc = (1.0 - gate) * self.lin_z_to_loc(z_tm1) + gate * prop_mean
        scale = self.softplus(self.lin_sig(self.relu(prop_mean))) + 1e-4
        return loc, scale


# ============================================================
# DynVAE model/guide (categorical a) + static covariates v
# ============================================================

class DynVAE_CatA(nn.Module):
    def __init__(
        self,
        dim_x=20,
        dim_z=3,
        num_actions=6,
        a_emb_dim=8,

        # static v
        dim_v=91,           # <-- requested
        v_emb_dim=5,
        v_hidden=128,

        dim_h=64,
        dim_g=64,
        transition_dim=64,
        hidden_dx=128,
        hidden_pa=64,
        hidden_y=64,
        num_layers=1,
        rnn_dropout=0.1,
        min_scale_x=0.1,
        max_scale_x=0.8,

        action_loss_weight=20.0, ######## 10-30
        action_class_weights=None,

        y_loss_weight=10.0, ######## 10-50
        y_class_weights=None,

        dim_f=64,
        hidden_qtilde=64,
        min_scale_z=1e-3,
        max_scale_z=5.0,

        device=None,
    ):
        super().__init__()
        self.dim_x = int(dim_x)
        self.dim_z = int(dim_z)
        self.K = int(num_actions)
        self.a_emb_dim = int(a_emb_dim)

        self.dim_v = int(dim_v)
        self.v_emb_dim = int(v_emb_dim)

        self.dim_h = int(dim_h)
        self.dim_g = int(dim_g)
        self.dim_f = int(dim_f)
        self.num_layers = int(num_layers)

        self.action_loss_weight = float(action_loss_weight)
        if action_class_weights is None:
            action_class_weights = torch.ones(self.K)
        self.register_buffer("action_class_weights", action_class_weights.float())

        self.y_loss_weight = float(y_loss_weight)
        if y_class_weights is None:
            y_class_weights = torch.ones(2)
        self.register_buffer("y_class_weights", y_class_weights.float())

        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # v encoder
        self.v_enc = MLPEmbed(in_dim=self.dim_v, out_dim=self.v_emb_dim, hidden_dim=v_hidden, dropout=0.0)

        # a embedding
        self.a_emb = nn.Embedding(self.K, self.a_emb_dim)

        # p(a_t | z_t, v)
        self.p_a = MLPCat(in_dim=self.dim_z + self.v_emb_dim, num_classes=self.K, hidden_dim=hidden_pa)

        # p(z_1)
        self.z1_loc = nn.Parameter(torch.zeros(self.dim_z))
        self.z1_unconstrained_scale = nn.Parameter(torch.zeros(self.dim_z))
        self.softplus = nn.Softplus()

        # p(z_t | z_{t-1}, a_{t-1}, v)
        self.p_z = GatedTransitionZCatAV(
            z_dim=self.dim_z,
            a_emb_dim=self.a_emb_dim,
            v_emb_dim=self.v_emb_dim,
            transition_dim=transition_dim,
        )

        # p(x_t | h_t)
        self.p_x = MLPNormal(in_dim=self.dim_h, out_dim=self.dim_x, hidden_dim=hidden_dx,
                             min_scale=min_scale_x, max_scale=max_scale_x)

        # p(y | z_T, a_T, v)
        self.p_y = MLPLogits(in_dim=self.dim_z + self.K + self.v_emb_dim, out_dim=1, hidden_dim=hidden_y)

        # generative GRU
        self.gen_rnn = nn.GRU(
            input_size=self.dim_z,
            hidden_size=self.dim_h,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )

        # smoother backward GRU over [x, emb(a), y, v_emb]
        self.inf_rnn_bw = nn.GRU(
            input_size=self.dim_x + self.a_emb_dim + 1 + self.v_emb_dim,
            hidden_size=self.dim_g,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=(rnn_dropout if self.num_layers > 1 else 0.0),
            bidirectional=False,
        )

        # distillation forward GRU over [x, emb(a_{t-1}), v_emb]
        self.inf_rnn_fw = nn.GRU(
            input_size=self.dim_x + self.a_emb_dim + self.v_emb_dim,
            hidden_size=self.dim_f,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=(rnn_dropout if self.num_layers > 1 else 0.0),
            bidirectional=False,
        )
        self.q_tilde = MLPNormal(
            in_dim=self.dim_f,
            out_dim=self.dim_z,
            hidden_dim=hidden_qtilde,
            min_scale=min_scale_z,
            max_scale=max_scale_z,
        )

        # q(z_t | [z_{t-1}, emb(a_{t-1}), v_emb], g_t)
        self.combiner = ContextCombiner(
            context_dim=self.dim_z + self.a_emb_dim + self.v_emb_dim,
            g_dim=self.dim_g,
            z_dim=self.dim_z,
        )

        # init states
        self.h0 = nn.Parameter(torch.zeros(self.num_layers, 1, self.dim_h))
        self.g0 = nn.Parameter(torch.zeros(self.num_layers, 1, self.dim_g))
        self.f0 = nn.Parameter(torch.zeros(self.num_layers, 1, self.dim_f))
        self.z_q0 = nn.Parameter(torch.zeros(self.dim_z))

        self.to(self.device)

    def p_z1_params(self, batch_size: int, device: torch.device):
        loc = self.z1_loc.unsqueeze(0).expand(batch_size, -1).to(device)
        scale = self.softplus(self.z1_unconstrained_scale).unsqueeze(0).expand(batch_size, -1).to(device) + 1e-4
        return loc, scale

    @torch.no_grad()
    def sample_z1(self, batch_size: int = 1):
        """Sample z_1 from p(z_1) for RL environment initialization.
        
        Returns:
            z1: (dim_z,) if batch_size=1, else (batch_size, dim_z)
        """
        loc, scale = self.p_z1_params(batch_size, self.device)
        z = torch.distributions.Normal(loc, scale).sample()
        return z.squeeze(0) if batch_size == 1 else z

    @torch.no_grad()
    def infer_z1_from_x1(self, x1: torch.Tensor, v: torch.Tensor, use_mean: bool = False):
        """从病人的初始观测 x1 和静态协变量 v 推断 z1（后验路径）。
        
        使用 q_tilde 路径（前向 RNN + q_tilde 网络），这是训练时使用的蒸馏后验。
        这个方法只需要 x1 和 v，不需要 a1 和 y。
        
        Args:
            x1: (dim_x,) 或 (batch_size, dim_x) - 病人的第一个时间步观测
            v: (dim_v,) 或 (batch_size, dim_v) - 病人的静态协变量
            use_mean: 如果 True，返回分布的均值；如果 False，从分布中采样
        
        Returns:
            z1: (dim_z,) 如果输入是单个样本，否则 (batch_size, dim_z)
            loc: (dim_z,) 或 (batch_size, dim_z) - z1 分布的均值
            scale: (dim_z,) 或 (batch_size, dim_z) - z1 分布的标准差
        """
        self.eval()
        
        # 处理输入维度
        x1 = x1.to(self.device)
        v = v.to(self.device)
        
        if x1.dim() == 1:
            x1 = x1.unsqueeze(0)  # (1, dim_x)
            single_sample = True
        else:
            single_sample = False
        
        if v.dim() == 1:
            v = v.unsqueeze(0)  # (1, dim_v)
        
        B = x1.shape[0]
        T = 1  # 只有第一个时间步
        
        # 编码 v
        v_emb = self.v_enc(v)  # (B, v_emb_dim)
        v_rep = v_emb.unsqueeze(1)  # (B, 1, v_emb_dim)
        
        # 构建输入：对于 t=1，a_shift[0] = 0（没有前一个 action）
        a_shift = torch.zeros(B, dtype=torch.long, device=self.device)  # (B,)
        a_shift_emb = self.a_emb(a_shift)  # (B, a_emb_dim)
        a_shift_emb = a_shift_emb.unsqueeze(1)  # (B, 1, a_emb_dim)
        
        # 拼接 [x1, a_shift_emb, v_rep]
        w = torch.cat([x1.unsqueeze(1), a_shift_emb, v_rep], dim=-1)  # (B, 1, dim_x + a_emb_dim + v_emb_dim)
        
        # 前向 RNN
        f_init = self.f0.expand(self.num_layers, B, self.dim_f).contiguous()
        f_seq, _ = self.inf_rnn_fw(w, f_init)  # (B, 1, dim_f)
        f_1 = f_seq[:, 0, :]  # (B, dim_f)
        
        # 通过 q_tilde 网络得到 z1 的分布参数
        loc, scale = self.q_tilde(f_1)  # 每个都是 (B, dim_z)
        scale = scale.clamp(min=1e-6)
        
        # 采样或返回均值
        if use_mean:
            z1 = loc
        else:
            z1 = torch.distributions.Normal(loc, scale).sample()
        
        if single_sample:
            return z1.squeeze(0), loc.squeeze(0), scale.squeeze(0)
        else:
            return z1, loc, scale

    def model(
        self,
        batch_x,
        batch_a,
        batch_y,
        batch_v,
        batch_mask,
        batch_lengths,
        annealing_factor: float = 1.0,
        distill_weight: float = 0.0,
    ):
        pyro.module("DynVAE_CatA", self)

        x = batch_x.to(self.device)
        a = batch_a.to(self.device)
        y = batch_y.to(self.device)
        v = batch_v.to(self.device)
        mask = batch_mask.to(self.device)
        lengths = batch_lengths.to(self.device)

        B, T_max, _ = x.shape

        v_emb = self.v_enc(v)
        v_rep = v_emb.unsqueeze(1).expand(B, T_max, self.v_emb_dim)

        # distillation forward features f_t (optional)
        f = None
        distill_weight = float(distill_weight)
        if distill_weight > 0.0:
            a_shift = torch.zeros_like(a)
            a_shift[:, 1:] = a[:, :-1]
            a_shift_emb = self.a_emb(a_shift)
            w = torch.cat([x, a_shift_emb, v_rep], dim=-1) * mask.unsqueeze(-1).float()

            f_init = self.f0.expand(self.num_layers, B, self.dim_f).contiguous()
            packed_w = pack_padded_sequence(w, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_f, _ = self.inf_rnn_fw(packed_w, f_init)
            f, _ = pad_packed_sequence(packed_f, batch_first=True, total_length=T_max)
            f = f * mask.unsqueeze(-1).float()

        h = self.h0.expand(self.num_layers, B, self.dim_h).contiguous()

        z_prev: Optional[torch.Tensor] = None
        a_prev_idx = torch.zeros(B, dtype=torch.long, device=self.device)
        a_prev_emb = self.a_emb(a_prev_idx)

        beta = max(float(annealing_factor), 1e-8)

        with pyro.plate("batch", B):
            for t in pyro.markov(range(T_max)):
                mask_t = mask[:, t]

                if t == 0:
                    z_loc, z_scale = self.p_z1_params(B, self.device)
                else:
                    z_loc, z_scale = self.p_z(z_prev, a_prev_emb, v_emb)

                with poutine.scale(scale=beta):
                    z_t = pyro.sample(f"z_{t+1}", dist.Normal(z_loc, z_scale).to_event(1).mask(mask_t))

                if f is not None:
                    zt_loc_d, zt_scale_d = self.q_tilde(f[:, t, :])
                    log_qtilde = dist.Normal(zt_loc_d, zt_scale_d).log_prob(z_t.detach()).sum(-1)
                    pyro.factor(f"distill_{t+1}", distill_weight * mask_t.float() * log_qtilde)

                # a_t observed
                a_logits = self.p_a(torch.cat([z_t, v_emb], dim=-1))
                w_t = self.action_class_weights[a[:, t]].clamp_min(1e-6)
                w_t = torch.where(mask_t, w_t, torch.ones_like(w_t))
                with poutine.scale(scale=self.action_loss_weight * w_t):
                    pyro.sample(f"a_{t+1}", dist.Categorical(logits=a_logits).mask(mask_t), obs=a[:, t])

                # update h only on valid
                out, h_new = self.gen_rnn(z_t.unsqueeze(1), h)
                h = torch.where(mask_t.view(1, B, 1), h_new, h)
                h_t = out[:, 0, :]

                # x_t observed
                x_loc, x_scale = self.p_x(h_t)
                pyro.sample(f"x_{t+1}", dist.Normal(x_loc, x_scale).to_event(1).mask(mask_t), obs=x[:, t, :])

                # last valid z/a
                if t == 0:
                    z_prev = z_t
                else:
                    z_prev = torch.where(mask_t.unsqueeze(-1), z_t, z_prev)
                a_prev_idx = torch.where(mask_t, a[:, t], a_prev_idx)
                a_prev_emb = self.a_emb(a_prev_idx)

            # y observed
            aT_onehot = F.one_hot(a_prev_idx, num_classes=self.K).float()
            y_logits = self.p_y(torch.cat([z_prev, aT_onehot, v_emb], dim=-1))
            y_idx = y.view(-1).long().clamp(0, 1)
            w_y = self.y_class_weights[y_idx].clamp_min(1e-6)
            with poutine.scale(scale=self.y_loss_weight * w_y):
                pyro.sample("y", dist.Bernoulli(logits=y_logits).to_event(1), obs=y)

    def guide(
        self,
        batch_x,
        batch_a,
        batch_y,
        batch_v,
        batch_mask,
        batch_lengths,
        annealing_factor: float = 1.0,
        distill_weight: float = 0.0,
    ):
        pyro.module("DynVAE_CatA", self)

        x = batch_x.to(self.device)
        a = batch_a.to(self.device)
        y = batch_y.to(self.device)
        v = batch_v.to(self.device)
        mask = batch_mask.to(self.device)
        lengths = batch_lengths.to(self.device)

        B, T_max, _ = x.shape

        v_emb = self.v_enc(v)
        v_rep = v_emb.unsqueeze(1).expand(B, T_max, self.v_emb_dim)

        a_emb_t = self.a_emb(a)
        y_rep = y.unsqueeze(1).expand(B, T_max, 1)
        u = torch.cat([x, a_emb_t, y_rep, v_rep], dim=-1) * mask.unsqueeze(-1).float()

        u_rev = reverse_padded_sequence(u, lengths)
        g_init = self.g0.expand(self.num_layers, B, self.dim_g).contiguous()

        packed = pack_padded_sequence(u_rev, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.inf_rnn_bw(packed, g_init)
        g_rev, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=T_max)
        g = reverse_padded_sequence(g_rev, lengths) * mask.unsqueeze(-1).float()

        z_prev = self.z_q0.expand(B, self.dim_z)
        a_prev_idx = torch.zeros(B, dtype=torch.long, device=self.device)
        a_prev_emb = self.a_emb(a_prev_idx)

        beta = max(float(annealing_factor), 1e-8)

        with pyro.plate("batch", B):
            for t in pyro.markov(range(T_max)):
                mask_t = mask[:, t]
                context = torch.cat([z_prev, a_prev_emb, v_emb], dim=-1)
                z_loc, z_scale = self.combiner(context, g[:, t, :])
                with poutine.scale(scale=beta):
                    z_t = pyro.sample(f"z_{t+1}", dist.Normal(z_loc, z_scale).to_event(1).mask(mask_t))

                z_prev = torch.where(mask_t.unsqueeze(-1), z_t, z_prev)
                a_prev_idx = torch.where(mask_t, a[:, t], a_prev_idx)
                a_prev_emb = self.a_emb(a_prev_idx)

    @torch.no_grad()
    def distill_kl_sum_count(self, x, a, y, v, mask, lengths, use_mean_z_prev: bool = True):
        """KL(q_phi || q_tilde) summed over valid time steps."""
        self.eval()
        x = x.to(self.device)
        a = a.to(self.device)
        y = y.to(self.device)
        v = v.to(self.device)
        mask = mask.to(self.device)
        lengths = lengths.to(self.device)

        B, T_max, _ = x.shape
        v_emb = self.v_enc(v)
        v_rep = v_emb.unsqueeze(1).expand(B, T_max, self.v_emb_dim)

        # g_t
        a_emb_t = self.a_emb(a)
        y_rep = y.unsqueeze(1).expand(B, T_max, 1)
        u = torch.cat([x, a_emb_t, y_rep, v_rep], dim=-1) * mask.unsqueeze(-1).float()

        u_rev = reverse_padded_sequence(u, lengths)
        g_init = self.g0.expand(self.num_layers, B, self.dim_g).contiguous()
        packed = pack_padded_sequence(u_rev, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.inf_rnn_bw(packed, g_init)
        g_rev, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=T_max)
        g = reverse_padded_sequence(g_rev, lengths) * mask.unsqueeze(-1).float()

        # f_t
        a_shift = torch.zeros_like(a)
        a_shift[:, 1:] = a[:, :-1]
        a_shift_emb = self.a_emb(a_shift)
        w = torch.cat([x, a_shift_emb, v_rep], dim=-1) * mask.unsqueeze(-1).float()

        f_init = self.f0.expand(self.num_layers, B, self.dim_f).contiguous()
        packed_w = pack_padded_sequence(w, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_f, _ = self.inf_rnn_fw(packed_w, f_init)
        f, _ = pad_packed_sequence(packed_f, batch_first=True, total_length=T_max)
        f = f * mask.unsqueeze(-1).float()

        loc_tilde, scale_tilde = self.q_tilde(f.reshape(-1, self.dim_f))
        loc_tilde = loc_tilde.reshape(B, T_max, self.dim_z)
        scale_tilde = scale_tilde.reshape(B, T_max, self.dim_z).clamp_min(1e-6)

        z_prev = self.z_q0.expand(B, self.dim_z)
        a_prev_idx = torch.zeros(B, dtype=torch.long, device=self.device)
        a_prev_emb = self.a_emb(a_prev_idx)

        kl_sum = x.new_zeros(())
        count = x.new_zeros(())

        for t in range(T_max):
            context = torch.cat([z_prev, a_prev_emb, v_emb], dim=-1)
            loc_phi, scale_phi = self.combiner(context, g[:, t, :])
            scale_phi = scale_phi.clamp_min(1e-6)

            q_phi_t = td.Normal(loc_phi, scale_phi)
            q_tilde_t = td.Normal(loc_tilde[:, t, :], scale_tilde[:, t, :])
            kl_bt = td.kl_divergence(q_phi_t, q_tilde_t).sum(-1)

            m = mask[:, t].float()
            kl_sum = kl_sum + (kl_bt * m).sum()
            count = count + m.sum()

            z_next = loc_phi if use_mean_z_prev else q_phi_t.rsample()
            z_prev = torch.where(mask[:, t].unsqueeze(-1), z_next, z_prev)
            a_prev_idx = torch.where(mask[:, t], a[:, t], a_prev_idx)
            a_prev_emb = self.a_emb(a_prev_idx)

        return kl_sum.detach(), count.detach()


# ============================================================
# KL annealing schedule
# ============================================================

def make_anneal_fn(steps_per_epoch, epochs, kl_start_epoch=0, ramp_fraction_of_remaining=0.5, eps=1e-8):
    total_steps = int(steps_per_epoch * epochs)
    start_step = int(steps_per_epoch * kl_start_epoch)
    remaining = max(0, total_steps - start_step)
    ramp_steps = max(1, int(ramp_fraction_of_remaining * remaining))

    def anneal(step):
        if step <= start_step:
            return float(eps)
        s = step - start_step
        return float(max(eps, min(1.0, s / float(ramp_steps))))

    info = {
        "total_steps": total_steps,
        "start_step": start_step,
        "ramp_steps": ramp_steps,
        "ramp_end_step": start_step + ramp_steps,
    }
    return anneal, info


# ============================================================
# Class weights (loader: x,a,y,v,lengths,mask)
# ============================================================

def compute_action_class_weights(train_loader, K: int):
    counts = torch.zeros(K)
    for _x, a, _y, _v, lengths, mask in train_loader:
        counts += torch.bincount(a[mask].flatten(), minlength=K).float()
    freq = counts / counts.sum().clamp_min(1.0)
    w = 1.0 / (freq + 1e-6)
    w = w / w.mean().clamp_min(1e-6)
    return w


def compute_binary_class_weights(train_loader):
    counts = torch.zeros(2)
    for _x, _a, y, _v, lengths, mask in train_loader:
        y_idx = y.view(-1).long().clamp(0, 1)
        counts += torch.bincount(y_idx, minlength=2).float()
    freq = counts / counts.sum().clamp_min(1.0)
    w = 1.0 / (freq + 1e-6)
    w = w / w.mean().clamp_min(1e-6)
    return w


# ============================================================
# Training
# ============================================================

def train_svi(
    train_loader,
    model: DynVAE_CatA,
    test_loader=None,
    epochs=80,
    lr=1e-3,
    kl_start_epoch=2,
    seed=0,
    log_every=1,
    num_particles=1,
    distill_weight=1.0,
):
    pyro.set_rng_seed(seed)
    pyro.clear_param_store()

    optim = ClippedAdam({"lr": lr, "clip_norm": 5.0})
    loss_fn = Trace_ELBO(num_particles=num_particles, vectorize_particles=(num_particles > 1))
    svi = SVI(model.model, model.guide, optim, loss=loss_fn)

    steps_per_epoch = len(train_loader)
    anneal_fn, info = make_anneal_fn(
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        kl_start_epoch=kl_start_epoch,
        ramp_fraction_of_remaining=0.5,
        eps=1e-8,
    )
    print(
        f"[info] steps/epoch={steps_per_epoch}, total_steps={info['total_steps']}, "
        f"kl_start_epoch={kl_start_epoch}, start_step={info['start_step']}, "
        f"ramp_steps={info['ramp_steps']}, ramp_end_step={info['ramp_end_step']}"
    )

    def _print_header():
        print("epoch |  train loss/t | test@1.0 loss/t |  beta || distill KL/t (q_phi || q_tilde)")
        print("----- | ------------ | -------------- | ----- || ---------------------------")

    step = 0
    beta = 1.0
    for epoch in range(1, epochs + 1):
        model.train()
        for x, a, y, v, lengths, mask in train_loader:
            step += 1
            beta = anneal_fn(step)
            svi.step(x, a, y, v, mask, lengths, annealing_factor=beta, distill_weight=distill_weight)

        model.eval()
        train_elbo = 0.0
        train_T = 0.0
        distill_kl_sum = 0.0
        distill_kl_count = 0.0

        with torch.no_grad():
            for x, a, y, v, lengths, mask in train_loader:
                train_elbo += float(
                    svi.evaluate_loss(x, a, y, v, mask, lengths, annealing_factor=beta, distill_weight=0.0)
                )
                train_T += float(lengths.sum().item())

                if distill_weight > 0.0:
                    kl_sum, kl_count = model.distill_kl_sum_count(x, a, y, v, mask, lengths)
                    distill_kl_sum += float(kl_sum.item())
                    distill_kl_count += float(kl_count.item())

        train_loss_per_t = train_elbo / max(train_T, 1.0)
        distill_kl_per_t = distill_kl_sum / max(distill_kl_count, 1.0)

        test_full_t = None
        if test_loader is not None:
            total_full = 0.0
            total_full_T = 0.0
            with torch.no_grad():
                for x, a, y, v, lengths, mask in test_loader:
                    total_full += float(
                        svi.evaluate_loss(x, a, y, v, mask, lengths, annealing_factor=1.0, distill_weight=0.0)
                    )
                    total_full_T += float(lengths.sum().item())
            test_full_t = total_full / max(total_full_T, 1.0)

        if epoch % log_every == 0:
            if epoch == 1 or epoch % 10 == 0:
                _print_header()
            if test_loader is None:
                print(f"{epoch:5d} | {train_loss_per_t:12.4f} | {'-':14s} | {beta:5.3f} || {distill_kl_per_t:27.4f}")
            else:
                print(f"{epoch:5d} | {train_loss_per_t:12.4f} | {test_full_t:14.4f} | {beta:5.3f} || {distill_kl_per_t:27.4f}")


# ============================================================
# Dataset / loaders (no z)
# ============================================================

class TrajectoryDataset(Dataset):
    """
    Each item:
      x: (T,dim_x) float
      a: (T,) long
      y: (1,) float
      v: (dim_v,) float
    """
    def __init__(self, trajectories: Sequence[Dict[str, Any]]):
        self.trajs = list(trajectories)

    def __len__(self) -> int:
        return len(self.trajs)

    def __getitem__(self, idx: int):
        tr = self.trajs[idx]
        x = torch.as_tensor(tr["x"], dtype=torch.float32)
        a = torch.as_tensor(tr["a"], dtype=torch.long)
        y = torch.as_tensor(tr["y"], dtype=torch.float32).view(1)
        v = torch.as_tensor(tr["v"], dtype=torch.float32)
        return x, a, y, v


def collate_trajectories(batch: List[Tuple[torch.Tensor, ...]]):
    xs, aas, ys, vs = zip(*batch)

    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    T_max = int(lengths.max().item())

    x_pad = pad_sequence(xs, batch_first=True, padding_value=0.0)  # (B,T,dim_x)
    a_pad = pad_sequence(aas, batch_first=True, padding_value=0)   # (B,T)

    mask = (torch.arange(T_max)[None, :] < lengths[:, None])       # (B,T) bool

    v_stack = torch.stack(vs, dim=0)                               # (B,dim_v)
    y_stack = torch.stack(ys, dim=0)                               # (B,1)

    return x_pad, a_pad, y_stack, v_stack, lengths, mask


def make_toy_dataset(
    n: int,
    dim_x: int,
    dim_v: int,
    num_actions: int,
    min_T: int = 10,
    max_T: int = 30,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    trajs: List[Dict[str, Any]] = []
    for _ in range(n):
        T = int(rng.integers(min_T, max_T + 1))
        x = rng.normal(size=(T, dim_x)).astype(np.float32)
        a = rng.integers(0, num_actions, size=(T,), dtype=np.int64)
        y = rng.integers(0, 2, size=(1,), dtype=np.int64).astype(np.float32)
        v = rng.normal(size=(dim_v,)).astype(np.float32)
        trajs.append({"x": x, "a": a, "y": y, "v": v})
    return trajs


def build_loaders_from_npz(npz_path: str, batch_size: int, num_workers: int = 0):
    """
    Expected arrays in npz:
      - x: (N,T,dim_x) padded
      - a: (N,T) padded
      - y: (N,1)
      - v: (N,dim_v)
      - lengths: (N,)
    """
    data = np.load(npz_path)
    x = torch.tensor(data["x"], dtype=torch.float32)
    a = torch.tensor(data["a"], dtype=torch.long)
    y = torch.tensor(data["y"], dtype=torch.float32)
    v = torch.tensor(data["v"], dtype=torch.float32)
    lengths = torch.tensor(data["lengths"], dtype=torch.long)

    T_max = x.shape[1]
    mask = (torch.arange(T_max)[None, :] < lengths[:, None])

    class _Padded(Dataset):
        def __len__(self): return x.shape[0]
        def __getitem__(self, i):
            return x[i], a[i], y[i], v[i], lengths[i], mask[i]

    ds = _Padded()
    n_test = max(1, int(0.2 * len(ds)))
    n_train = len(ds) - n_test
    train_ds, test_ds = random_split(ds, [n_train, n_test], generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


# ============================================================
# Main (notebook-safe argparse)
# ============================================================

def main(argv=None):
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--kl_start_epoch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)

    # NOTEBOOK SAFE: ignore extra jupyter args like -f kernel.json
    if argv is None:
        args, _ = parser.parse_known_args()
    else:
        args, _ = parser.parse_known_args(argv)

    set_all_seeds(args.seed)

    # match your setup (edit if needed)
    dim_x = 39
    dim_z = 6
    num_actions = 7
    dim_v = 91

    train_loader, test_loader, report = create_loaders(
        use_incremental_for_cumulative_x=True,
        pid_width=None,
        normalize_x=True,
        normalize_v=True,
        scale_nonneg_only=True,
        plot_length_hist=True,
        asa_encoding="onehot",  # or "ordinal"
    )

    a_w = compute_action_class_weights(train_loader, K=num_actions)
    y_w = compute_binary_class_weights(train_loader)

    model = DynVAE_CatA(
        dim_x=dim_x,
        dim_z=dim_z,
        num_actions=num_actions,
        a_emb_dim=10,

        dim_v=dim_v,
        v_emb_dim=5,
        v_hidden=128,

        dim_h=64,
        dim_g=64,
        transition_dim=64,
        hidden_dx=64,
        hidden_pa=64,
        hidden_y=64,
        num_layers=1,
        rnn_dropout=0.1,
        min_scale_x=0.10,
        max_scale_x=0.80,

        action_loss_weight=10.0,
        action_class_weights=a_w,

        y_loss_weight=10.0,
        y_class_weights=y_w,

        dim_f=64,
        hidden_qtilde=64,
        min_scale_z=1e-3,
        max_scale_z=5.0,
    )

    train_svi(
        train_loader=train_loader,
        model=model,
        test_loader=test_loader,
        epochs=args.epochs,
        lr=args.lr,
        kl_start_epoch=args.kl_start_epoch,
        seed=args.seed,
        log_every=1,
        num_particles=1,
        distill_weight=1.0,
    )


# If running as a script, this works; in notebooks, just call main([])
if __name__ == "__main__":
    main()
