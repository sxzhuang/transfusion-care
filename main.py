from __future__ import annotations

import argparse
import copy
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from dynvae_model import DynVAE_CatA
from env import TransfusionEnv
from load_data import SurgerySequence, create_loaders
from network import PPOAgent
from ppo_algorithm import PPOAlgorithm
from train_rl import DynVAE_Adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppo_actor_lr", type=float, default=1e-3)
    parser.add_argument("--ppo_critic_lr", type=float, default=3e-3)
    parser.add_argument("--ppo_gamma", type=float, default=0.99)
    parser.add_argument("--ppo_lambda", type=float, default=0.95)
    parser.add_argument("--ppo_clip", type=float, default=0.2)
    parser.add_argument("--ppo_epochs", type=int, default=10)
    parser.add_argument("--ppo_batch_size", type=int, default=32)
    parser.add_argument("--ppo_hidden_dim", type=int, default=128)
    parser.add_argument("--entropy_coef", type=float, default=1e-2)
    parser.add_argument("--total_steps", type=int, default=20000)
    parser.add_argument("--rollout_steps", type=int, default=128)
    parser.add_argument("--max_T", type=int, default=5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument(
        "--reward_mode",
        type=str,
        choices=["base", "fixed_potential", "critic_based"],
        default="base",
    )
    parser.add_argument("--potential_mc_samples", type=int, default=10)
    parser.add_argument("--kl_coef", type=float, default=0.05)
    parser.add_argument("--target_tau", type=float, default=0.005)
    parser.add_argument("--eval_interval", type=int, default=2048)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--eval_num_envs", type=int, default=None)
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument(
        "--model_path",
        type=str,
        default=(
            "/slurm/rongfeng/icml_aki/real_data/experiments/"
            "20260125_012803/checkpoint_epoch_0060.pt"
        ),
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def build_run_name(args: argparse.Namespace) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"rl_{timestamp}_seed{args.seed}_{args.reward_mode}"
    if args.run_name:
        return args.run_name.strip().replace(" ", "_")
    return base


def split_sequences(
    sequences: Sequence[SurgerySequence],
    seed: int,
    test_y0: int = 50,
    test_y1: int = 50,
) -> tuple[list[SurgerySequence], list[SurgerySequence]]:
    rng = np.random.RandomState(seed)
    y0_indices = [i for i, seq in enumerate(sequences) if int(seq.y) == 0]
    y1_indices = [i for i, seq in enumerate(sequences) if int(seq.y) == 1]
    if len(y0_indices) < test_y0 or len(y1_indices) < test_y1:
        raise ValueError("Not enough samples to build the requested test split.")
    test_indices = np.concatenate(
        [
            rng.choice(y0_indices, size=test_y0, replace=False),
            rng.choice(y1_indices, size=test_y1, replace=False),
        ]
    )
    test_set = {int(i) for i in test_indices}
    test_sequences = [sequences[i] for i in test_indices]
    train_sequences = [seq for i, seq in enumerate(sequences) if i not in test_set]
    rng.shuffle(test_sequences)
    rng.shuffle(train_sequences)
    return train_sequences, test_sequences


def load_sequences(seed: int) -> tuple[list[SurgerySequence], list[SurgerySequence]]:
    train_loader, _, _ = create_loaders(plot_length_hist=False, seed=seed)
    dataset = train_loader.dataset.dataset
    sequences = dataset.sequences
    return split_sequences(sequences, seed)


def load_checkpoint_model(checkpoint_path: str, device: torch.device) -> DynVAE_CatA:
    print(f"加载模型从: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint.get("model_config", {})

    dim_x = model_config.get("dim_x", 39)
    dim_z = model_config.get("dim_z", 6)
    num_actions = model_config.get("num_actions", 7)
    dim_v = model_config.get("dim_v", 91)
    hidden_dx = model_config.get("hidden_dx", 128)
    hidden_pa = model_config.get("hidden_pa", 64)
    hidden_y = model_config.get("hidden_y", 64)

    print(f"模型配置: dim_x={dim_x}, dim_z={dim_z}, num_actions={num_actions}, dim_v={dim_v}")

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
        hidden_dx=hidden_dx,
        hidden_pa=hidden_pa,
        hidden_y=hidden_y,
        num_layers=1,
        rnn_dropout=0.1,
        min_scale_x=0.10,
        max_scale_x=0.80,
        action_loss_weight=20.0,
        y_loss_weight=20.0,
        dim_f=64,
        hidden_qtilde=64,
        min_scale_z=1e-3,
        max_scale_z=5.0,
        device=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def offline_policy_logits(adapter: DynVAE_Adapter, z_t: torch.Tensor) -> torch.Tensor:
    base_model = adapter.base_model
    device = base_model.device
    z_t = z_t.to(device)
    v_emb = adapter.default_v_emb
    if z_t.dim() == 1:
        z_t = z_t.unsqueeze(0)
    if v_emb.dim() == 1:
        v_emb = v_emb.unsqueeze(0)
    return base_model.p_a(torch.cat([z_t, v_emb], dim=-1))


@torch.no_grad()
def offline_policy_logprob(adapter: DynVAE_Adapter, z_t: torch.Tensor, action_index: int) -> float:
    logits = offline_policy_logits(adapter, z_t)
    dist = torch.distributions.Categorical(logits=logits)
    action_tensor = torch.tensor([int(action_index)], device=logits.device)
    return float(dist.log_prob(action_tensor).squeeze(0).item())


@torch.no_grad()
def offline_policy_sample(adapter: DynVAE_Adapter, z_t: torch.Tensor) -> int:
    logits = offline_policy_logits(adapter, z_t)
    dist = torch.distributions.Categorical(logits=logits)
    return int(dist.sample().squeeze(0).item())


@torch.no_grad()
def sample_next_z(adapter: DynVAE_Adapter, z_t: torch.Tensor, action_index: int) -> torch.Tensor:
    base_model = adapter.base_model
    device = base_model.device
    z_t = z_t.to(device)
    a_idx = torch.tensor([int(action_index)], device=device)
    a_emb = base_model.a_emb(a_idx)
    v_emb = adapter.default_v_emb.unsqueeze(0)
    loc, scale = base_model.p_z(z_t.unsqueeze(0), a_emb, v_emb)
    scale = torch.clamp(scale, min=1e-6)
    z_next = torch.distributions.Normal(loc, scale).sample()
    return z_next.squeeze(0)


@torch.no_grad()
def terminal_risk(adapter: DynVAE_Adapter, z_t: torch.Tensor, action_index: int) -> float:
    base_model = adapter.base_model
    device = base_model.device
    z_t = z_t.to(device)
    v_emb = adapter.default_v_emb
    if z_t.dim() == 1:
        z_t = z_t.unsqueeze(0)
    if v_emb.dim() == 1:
        v_emb = v_emb.unsqueeze(0)
    a_onehot = F.one_hot(
        torch.tensor([int(action_index)], device=device),
        num_classes=base_model.K,
    ).float()
    logits = base_model.p_y(torch.cat([z_t, a_onehot, v_emb], dim=-1))
    risk = torch.sigmoid(logits).squeeze(0).squeeze(0)
    return float(risk.item())


@torch.no_grad()
def compute_potential_mc(
    adapter: DynVAE_Adapter,
    z_t: torch.Tensor,
    start_t: int,
    max_T: int,
    num_samples: int,
) -> float:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    total = 0.0
    for _ in range(num_samples):
        z = z_t
        action = None
        for s in range(int(start_t), int(max_T) + 1):
            action = offline_policy_sample(adapter, z)
            if s < max_T:
                z = sample_next_z(adapter, z, action)
        if action is None:
            action = 0
        total += -terminal_risk(adapter, z, action)
    return total / float(num_samples)


def start_episode(
    adapter: DynVAE_Adapter,
    env: TransfusionEnv,
    sequence: SurgerySequence,
    device: torch.device,
    use_mean: bool,
) -> torch.Tensor:
    env.max_T = int(sequence.length)
    v = torch.tensor(sequence.v, dtype=torch.float32, device=device)
    x1 = torch.tensor(sequence.x[0], dtype=torch.float32, device=device)
    with torch.no_grad():
        v_batch = v.unsqueeze(0)
        v_emb = adapter.base_model.v_enc(v_batch).squeeze(0)
        adapter.default_v_emb.copy_(v_emb)
        z1, _, _ = adapter.base_model.infer_z1_from_x1(x1, v, use_mean=use_mean)
    obs, _ = env.reset(start_t=1, z_t=z1)
    return torch.tensor(obs, dtype=torch.float32, device=device)


def evaluate(
    agent: PPOAgent,
    adapters: Sequence[DynVAE_Adapter],
    envs: Sequence[TransfusionEnv],
    sequences: Sequence[SurgerySequence],
    device: torch.device,
    eval_episodes: int,
    use_mean: bool,
) -> dict[str, float]:
    y1_sum = 0.0
    y0_sum = 0.0
    total_sum = 0.0
    y0_pred0 = 0.0
    y1_pred0 = 0.0
    y0_total = 0.0
    y1_total = 0.0
    num_envs = len(envs)

    for sequence in sequences:
        probs = []
        remaining = int(eval_episodes)
        while remaining > 0:
            active = min(num_envs, remaining)
            obs_list = []
            done_flags = [False] * active
            for i in range(active):
                obs_list.append(start_episode(adapters[i], envs[i], sequence, device, use_mean))
            while not all(done_flags):
                for i in range(active):
                    if done_flags[i]:
                        continue
                    with torch.no_grad():
                        _, _, _, _, action_index = agent.get_action_and_value(obs_list[i])
                    next_obs, _, terminated, _, info = envs[i].step(action_index)
                    if terminated:
                        aki_prob = info.get("aki_prob")
                        if aki_prob is None:
                            raise RuntimeError("Episode terminated without aki_prob in info.")
                        probs.append(float(aki_prob))
                        pred0 = float(aki_prob) <= 0.5
                        if int(sequence.y) == 0:
                            y0_total += 1.0
                            if pred0:
                                y0_pred0 += 1.0
                        else:
                            y1_total += 1.0
                            if pred0:
                                y1_pred0 += 1.0
                        done_flags[i] = True
                    else:
                        obs_list[i] = torch.tensor(next_obs, dtype=torch.float32, device=device)
            remaining -= active

        mean_prob = float(np.mean(probs)) if probs else 0.0
        if int(sequence.y) == 1:
            y1_sum += 1.0 - mean_prob
        else:
            y0_sum += 0.0 - mean_prob
        total_sum += float(sequence.y) - mean_prob

    y0_pred0_rate = y0_pred0 / y0_total if y0_total > 0 else 0.0
    y1_pred0_rate = y1_pred0 / y1_total if y1_total > 0 else 0.0
    return {
        "y1_sum": y1_sum,
        "y0_sum": y0_sum,
        "total_sum": total_sum,
        "y0_pred0_rate": y0_pred0_rate,
        "y1_pred0_rate": y1_pred0_rate,
    }


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    base_model = load_checkpoint_model(args.model_path, device)
    num_envs = int(args.num_envs)
    eval_num_envs = int(args.eval_num_envs) if args.eval_num_envs is not None else num_envs

    run_name = build_run_name(args)
    args.run_name = run_name
    checkpoint_dir = Path("results") / run_name / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to: {checkpoint_dir}")

    train_adapters = [DynVAE_Adapter(base_model, default_v=None) for _ in range(num_envs)]
    eval_adapters = [DynVAE_Adapter(base_model, default_v=None) for _ in range(eval_num_envs)]
    for adapter in train_adapters + eval_adapters:
        adapter.eval()

    dose_values = np.linspace(0.0, 1.0, base_model.K)
    envs = [
        TransfusionEnv(adapter, max_T=args.max_T, beta=args.beta, dose_values=dose_values)
        for adapter in train_adapters
    ]
    eval_envs = [
        TransfusionEnv(adapter, max_T=args.max_T, beta=args.beta, dose_values=dose_values)
        for adapter in eval_adapters
    ]

    train_sequences, test_sequences = load_sequences(args.seed)
    rng = np.random.RandomState(args.seed)

    agent = PPOAgent(
        input_dim=base_model.dim_z,
        action_dim=base_model.K,
        hidden_dim=args.ppo_hidden_dim,
        device=device,
    )
    algorithm = PPOAlgorithm(
        learning_rate=args.ppo_actor_lr,
        critic_learning_rate=args.ppo_critic_lr,
        gamma=args.ppo_gamma,
        gae_lambda=args.ppo_lambda,
        clip_coef=args.ppo_clip,
        update_epochs=args.ppo_epochs,
        minibatch_size=args.ppo_batch_size,
        entropy_coef=args.entropy_coef,
        kl_coef=args.kl_coef,
        device=device,
    )

    target_critic = None
    if args.reward_mode == "critic_based":
        target_critic = copy.deepcopy(agent.critic)
        target_critic.to(device)
        for param in target_critic.parameters():
            param.requires_grad_(False)

    if not train_sequences:
        raise RuntimeError("Training split is empty.")
    if not test_sequences:
        raise RuntimeError("Balanced test split is empty.")

    def sample_train_sequence() -> SurgerySequence:
        idx = int(rng.randint(len(train_sequences)))
        return train_sequences[idx]

    obs_list = []
    for i in range(num_envs):
        current_sequence = sample_train_sequence()
        obs_list.append(start_episode(train_adapters[i], envs[i], current_sequence, device, use_mean=False))

    step = 0
    next_eval_step = int(args.eval_interval)
    while step < args.total_steps:
        batch = []
        while len(batch) < args.rollout_steps and step < args.total_steps:
            for i in range(num_envs):
                if len(batch) >= args.rollout_steps or step >= args.total_steps:
                    break
                obs_t = obs_list[i]
                with torch.no_grad():
                    _, logprob, _, value, action_index = agent.get_action_and_value(obs_t)

                ref_logprob = None
                if args.kl_coef > 0.0:
                    ref_logprob = offline_policy_logprob(train_adapters[i], obs_t, action_index)

                next_obs, reward, terminated, _, info = envs[i].step(action_index)
                done = bool(terminated)
                next_obs_t = None

                if done:
                    next_value = 0.0
                else:
                    next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)
                    with torch.no_grad():
                        next_value = float(agent.value(next_obs_t).item())

                if args.ablation:
                    if done:
                        aki_prob = info.get("aki_prob")
                        if aki_prob is None:
                            raise RuntimeError("Episode terminated without aki_prob in info.")
                        reward = -float(aki_prob)
                    else:
                        reward = 0.0
                else:
                    if args.reward_mode == "fixed_potential":
                        current_t = int(info["t"])
                        max_T = int(envs[i].max_T)
                        phi_t = compute_potential_mc(
                            train_adapters[i],
                            obs_t,
                            current_t,
                            max_T,
                            args.potential_mc_samples,
                        )
                        if done:
                            phi_next = 0.0
                        else:
                            phi_next = compute_potential_mc(
                                train_adapters[i],
                                next_obs_t,
                                current_t + 1,
                                max_T,
                                args.potential_mc_samples,
                            )
                        reward = float(reward) + args.ppo_gamma * phi_next - phi_t
                    elif args.reward_mode == "critic_based":
                        if target_critic is None:
                            raise RuntimeError("critic_based reward requires a target critic.")
                        with torch.no_grad():
                            phi_t = float(target_critic(obs_t).item())
                            if done:
                                phi_next = 0.0
                            else:
                                phi_next = float(target_critic(next_obs_t).item())
                        reward = float(reward) + args.ppo_gamma * phi_next - phi_t

                batch.append(
                    {
                        "obs": obs_t.detach().cpu(),
                        "action_index": int(action_index),
                        "logprob": float(logprob.item()),
                        "reward": float(reward),
                        "done": float(done),
                        "value": float(value.item()),
                        "next_value": float(next_value),
                        "legal_mask": None,
                        "t": int(info["t"]),
                        "ref_logprob": ref_logprob,
                    }
                )

                if done:
                    current_sequence = sample_train_sequence()
                    obs_list[i] = start_episode(
                        train_adapters[i],
                        envs[i],
                        current_sequence,
                        device,
                        use_mean=False,
                    )
                else:
                    obs_list[i] = next_obs_t
                step += 1

        metrics = algorithm.update(batch, agent)
        if metrics:
            print(
                "step={step} policy_loss={policy_loss:.4f} value_loss={value_loss:.4f} "
                "entropy={entropy:.4f} approx_kl={approx_kl:.4f}".format(
                    step=step,
                    **metrics,
                )
            )
        if args.reward_mode == "critic_based" and target_critic is not None:
            with torch.no_grad():
                for target_param, param in zip(target_critic.parameters(), agent.critic.parameters()):
                    target_param.data.mul_(1.0 - args.target_tau).add_(param.data, alpha=args.target_tau)
        if step >= next_eval_step:
            eval_metrics = evaluate(
                agent,
                eval_adapters,
                eval_envs,
                test_sequences,
                device,
                args.eval_episodes,
                use_mean=False,
            )
            print(
                "eval_step={step} y1_sum={y1_sum:.4f} y0_sum={y0_sum:.4f} "
                "total_sum={total_sum:.4f} y0_pred0_rate={y0_pred0_rate:.4f} "
                "y1_pred0_rate={y1_pred0_rate:.4f}".format(
                    step=step,
                    **eval_metrics,
                )
            )
            checkpoint_path = checkpoint_dir / f"checkpoint_step_{step:06d}.pt"
            torch.save(
                {
                    "step": step,
                    "agent_state_dict": agent.state_dict(),
                    "run_name": run_name,
                    "args": vars(args),
                },
                checkpoint_path,
            )
            print(f"Saved RL checkpoint to: {checkpoint_path}")
            next_eval_step += int(args.eval_interval)
            obs_list = []
            for i in range(num_envs):
                current_sequence = sample_train_sequence()
                obs_list.append(start_episode(train_adapters[i], envs[i], current_sequence, device, use_mean=False))


if __name__ == "__main__":
    main()
