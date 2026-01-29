#!/usr/bin/env python3
"""
强化学习训练主函数
使用训练好的 DynVAE_CatA 模型作为转移方程，训练 PPO 智能体

使用方法:
    # 基本用法（使用零向量作为默认 v）
    python train_rl.py --model_path path/to/model.pth
    
    # 使用训练数据的平均 v（推荐）
    python train_rl.py --model_path path/to/model.pth --use_avg_v
    
    # 自定义训练参数
    python train_rl.py --model_path path/to/model.pth --use_avg_v \
        --total_steps 5000 --ppo_actor_lr 1e-3 --max_T 5 --beta 0.1
    
    # 保存训练好的智能体
    python train_rl.py --model_path path/to/model.pth --use_avg_v \
        --save_agent path/to/agent.pth

参数说明:
    --model_path: 训练好的 DynVAE_CatA 模型检查点路径（必需）
    --use_avg_v: 使用训练数据的平均静态协变量 v（推荐）
    --total_steps: 总训练步数（默认: 2000）
    --rollout_steps: 每次 rollout 的步数（默认: 256）
    --max_T: 每个 episode 的最大时间步（默认: 5）
    --beta: 动作成本权重（默认: 0.1）
    --ppo_actor_lr: PPO actor 学习率（默认: 1e-3）
    --ppo_critic_lr: PPO critic 学习率（默认: 3e-3）
    --ppo_gamma: 折扣因子（默认: 0.99）
    --ppo_lambda: GAE lambda（默认: 0.95）
    --ppo_clip: PPO clip 系数（默认: 0.2）
    --ppo_epochs: 每次更新的 epoch 数（默认: 10）
    --ppo_batch_size: PPO minibatch 大小（默认: 32）
    --ppo_hidden_dim: PPO 网络隐藏层维度（默认: 128）
    --entropy_coef: 熵系数（默认: 1e-2）
    --seed: 随机种子（默认: 0）
    --device: 设备 cuda/cpu（默认: 自动选择）
    --save_agent: 保存训练好的智能体路径（可选）
    --log_every: 每 N 步打印一次日志（默认: 100）
"""
import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 添加 transfusion_care 目录到路径
transfusion_care_path = os.path.join(os.path.dirname(__file__), '..', 'transfusion_care')
if os.path.exists(transfusion_care_path):
    sys.path.insert(0, transfusion_care_path)

from dynvae_model import DynVAE_CatA, compute_action_class_weights, compute_binary_class_weights
from load_data import create_loaders

# 从 transfusion_care 导入
try:
    from env import TransfusionEnv
    from network import PPOAgent
    from ppo_algorithm import PPOAlgorithm
except ImportError as e:
    print(f"错误: 无法导入 transfusion_care 模块: {e}")
    print(f"请确保 {transfusion_care_path} 目录存在且包含 env.py, network.py, ppo_algorithm.py")
    sys.exit(1)


class DynVAE_Adapter(nn.Module):
    """
    适配器类：将 real_data 的 DynVAE_CatA 适配到 README.md 定义的接口
    
    real_data 的模型需要 v_emb（静态协变量），但强化学习接口不需要。
    我们使用训练数据的平均 v 来生成默认的 v_emb。
    """
    def __init__(self, base_model: DynVAE_CatA, default_v: torch.Tensor = None):
        super().__init__()
        self.base_model = base_model
        self.K = base_model.K
        self.dim_z = base_model.dim_z
        self.device = base_model.device
        self.a_emb = base_model.a_emb
        
        # 如果没有提供 default_v，使用零向量
        if default_v is None:
            default_v = torch.zeros(base_model.dim_v, device=self.device)
        else:
            default_v = default_v.to(self.device)
        
        # 计算并缓存默认的 v_emb
        with torch.no_grad():
            self.register_buffer('default_v_emb', base_model.v_enc(default_v.unsqueeze(0)).squeeze(0))
    
    def p_z(self, z: torch.Tensor, a_emb: torch.Tensor):
        """
        转移模型 p(z_{t+1} | z_t, a_t)
        
        Args:
            z: (B, dim_z)
            a_emb: (B, a_emb_dim)
        
        Returns:
            loc: (B, dim_z)
            scale: (B, dim_z)
        """
        # 扩展 v_emb 到批次大小
        B = z.shape[0]
        v_emb = self.default_v_emb.unsqueeze(0).expand(B, -1)
        
        # 调用 base_model 的 p_z（需要 v_emb）
        return self.base_model.p_z(z, a_emb, v_emb)
    
    def p_y(self, z_a_concat: torch.Tensor):
        """
        终端风险模型 p(y=1 | z_T, a_T)
        
        Args:
            z_a_concat: (B, dim_z + K) - concat([z_T, onehot(a_T)])
        
        Returns:
            logits: (B, 1)
        """
        B = z_a_concat.shape[0]
        z_T = z_a_concat[:, :self.dim_z]
        a_T_onehot = z_a_concat[:, self.dim_z:]
        
        # 扩展 v_emb 到批次大小
        v_emb = self.default_v_emb.unsqueeze(0).expand(B, -1)
        
        # base_model 的 p_y 需要 (z_T, a_T_onehot, v_emb)
        input_concat = torch.cat([z_T, a_T_onehot, v_emb], dim=-1)
        return self.base_model.p_y(input_concat)

    @torch.no_grad()
    def sample_z1(self):
        """
        初始潜在状态采样器
        
        Returns:
            z_t: (dim_z,)
        """
        return self.base_model.sample_z1(batch_size=1).squeeze(0)


def compute_average_v(train_loader, device):
    """
    从训练数据中计算平均的静态协变量 v
    """
    all_v = []
    for x, a, y, v, lengths, mask in train_loader:
        all_v.append(v)
    
    if len(all_v) == 0:
        return torch.zeros(91, device=device)  # 默认 dim_v=91
    
    v_stack = torch.cat(all_v, dim=0)  # (N, dim_v)
    avg_v = v_stack.mean(dim=0)  # (dim_v,)
    return avg_v.to(device)


def load_model(checkpoint_path, device, train_loader=None):
    """
    加载训练好的模型
    
    Args:
        checkpoint_path: 模型检查点路径
        device: 设备
        train_loader: 训练数据加载器（用于计算平均 v，可选）
    
    Returns:
        adapter: DynVAE_Adapter 实例
    """
    print(f"加载模型从: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint.get('model_config', {})
    
    # 从配置中获取模型参数
    dim_x = model_config.get('dim_x', 39)
    dim_z = model_config.get('dim_z', 6)
    num_actions = model_config.get('num_actions', 7)
    dim_v = model_config.get('dim_v', 91)
    
    print(f"模型配置: dim_x={dim_x}, dim_z={dim_z}, num_actions={num_actions}, dim_v={dim_v}")
    
    # 创建模型（使用默认参数，实际参数从 checkpoint 加载）
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
        action_loss_weight=20.0,
        y_loss_weight=20.0,
        dim_f=64,
        hidden_qtilde=64,
        min_scale_z=1e-3,
        max_scale_z=5.0,
        device=device,
    )
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✓ 模型加载成功")
    
    # 计算默认的 v（使用训练数据的平均值）
    if train_loader is not None:
        print("计算训练数据的平均静态协变量 v...")
        default_v = compute_average_v(train_loader, device)
        print(f"✓ 平均 v 计算完成 (shape: {default_v.shape})")
    else:
        default_v = None
        print("使用零向量作为默认 v")
    
    # 创建适配器
    adapter = DynVAE_Adapter(model, default_v=default_v)
    adapter.eval()
    
    return adapter


def parse_args():
    parser = argparse.ArgumentParser(description="使用训练好的 DynVAE 模型训练强化学习智能体")
    
    # 模型相关
    parser.add_argument("--model_path", type=str, required=True,
                        help="训练好的模型检查点路径")
    parser.add_argument("--use_avg_v", action="store_true",
                        help="使用训练数据的平均 v（需要加载数据）")
    
    # PPO 超参数
    parser.add_argument("--ppo_actor_lr", type=float, default=1e-3,
                        help="PPO actor 学习率")
    parser.add_argument("--ppo_critic_lr", type=float, default=3e-3,
                        help="PPO critic 学习率")
    parser.add_argument("--ppo_gamma", type=float, default=0.99,
                        help="折扣因子")
    parser.add_argument("--ppo_lambda", type=float, default=0.95,
                        help="GAE lambda")
    parser.add_argument("--ppo_clip", type=float, default=0.2,
                        help="PPO clip 系数")
    parser.add_argument("--ppo_epochs", type=int, default=10,
                        help="每次更新的 epoch 数")
    parser.add_argument("--ppo_batch_size", type=int, default=32,
                        help="PPO minibatch 大小")
    parser.add_argument("--ppo_hidden_dim", type=int, default=128,
                        help="PPO 网络隐藏层维度")
    parser.add_argument("--entropy_coef", type=float, default=1e-2,
                        help="熵系数")
    
    # 训练相关
    parser.add_argument("--total_steps", type=int, default=2000,
                        help="总训练步数")
    parser.add_argument("--rollout_steps", type=int, default=256,
                        help="每次 rollout 的步数")
    parser.add_argument("--max_T", type=int, default=5,
                        help="每个 episode 的最大时间步")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="动作成本权重")
    
    # 其他
    parser.add_argument("--seed", type=int, default=0,
                        help="随机种子")
    parser.add_argument("--device", type=str, default=None,
                        help="设备 (cuda/cpu)，默认自动选择")
    parser.add_argument("--save_agent", type=str, default=None,
                        help="保存训练好的智能体路径（可选）")
    parser.add_argument("--log_every", type=int, default=100,
                        help="每 N 步打印一次日志")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 设置设备
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print("="*60)
    print("强化学习训练脚本")
    print("="*60)
    print(f"设备: {device}")
    print(f"随机种子: {args.seed}")
    print("="*60)
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 加载训练数据（如果需要计算平均 v）
    train_loader = None
    if args.use_avg_v:
        print("\n加载训练数据以计算平均 v...")
        train_loader, _, _ = create_loaders(
            use_incremental_for_cumulative_x=True,
            pid_width=None,
            normalize_x=True,
            normalize_v=True,
            scale_nonneg_only=True,
            plot_length_hist=False,
            asa_encoding="onehot",
            batch_size_train=32,
            batch_size_test=64,
        )
        print("✓ 训练数据加载完成")
    
    # 加载模型
    print("\n" + "="*60)
    print("加载模型")
    print("="*60)
    adapter = load_model(args.model_path, device, train_loader if args.use_avg_v else None)
    
    # 创建环境
    print("\n" + "="*60)
    print("创建环境")
    print("="*60)
    dose_values = np.linspace(0.0, 1.0, adapter.K)
    env = TransfusionEnv(
        adapter,
        max_T=args.max_T,
        beta=args.beta,
        dose_values=dose_values,
    )
    print(f"✓ 环境创建完成")
    print(f"  动作空间: {adapter.K} 个离散动作")
    print(f"  状态空间维度: {adapter.dim_z}")
    print(f"  最大时间步: {args.max_T}")
    
    # 创建智能体
    print("\n" + "="*60)
    print("创建 PPO 智能体")
    print("="*60)
    agent = PPOAgent(
        input_dim=adapter.dim_z,
        action_dim=adapter.K,
        hidden_dim=args.ppo_hidden_dim,
        device=device,
    )
    print(f"✓ 智能体创建完成")
    
    # 创建 PPO 算法
    algorithm = PPOAlgorithm(
        learning_rate=args.ppo_actor_lr,
        critic_learning_rate=args.ppo_critic_lr,
        gamma=args.ppo_gamma,
        gae_lambda=args.ppo_lambda,
        clip_coef=args.ppo_clip,
        update_epochs=args.ppo_epochs,
        minibatch_size=args.ppo_batch_size,
        entropy_coef=args.entropy_coef,
        device=device,
    )
    
    # 训练循环
    print("\n" + "="*60)
    print("开始训练")
    print("="*60)
    
    obs, _ = env.reset(start_t=1, z_t=adapter.sample_z1())
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
    
    step = 0
    episode_count = 0
    episode_reward = 0.0
    
    while step < args.total_steps:
        batch = []
        for _ in range(args.rollout_steps):
            if step >= args.total_steps:
                break
            
            with torch.no_grad():
                _, logprob, _, value, action_index = agent.get_action_and_value(obs_t)
            
            next_obs, reward, terminated, _, info = env.step(action_index)
            done = bool(terminated)
            episode_reward += reward
            
            if done:
                next_value = 0.0
                next_obs, _ = env.reset(start_t=1, z_t=adapter.sample_z1())
                episode_count += 1
                if episode_count % 10 == 0:
                    print(f"Episode {episode_count}, 累计奖励: {episode_reward:.4f}")
                episode_reward = 0.0
            else:
                next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)
                with torch.no_grad():
                    next_value = float(agent.value(next_obs_t).item())
            
            batch.append({
                "obs": obs_t.detach().cpu(),
                "action_index": int(action_index),
                "logprob": float(logprob.item()),
                "reward": float(reward),
                "done": float(done),
                "value": float(value.item()),
                "next_value": float(next_value),
                "legal_mask": None,
                "t": int(info["t"]),
            })
            
            obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)
            step += 1
        
        # 更新策略
        metrics = algorithm.update(batch, agent)
        
        # 打印日志
        if metrics and step % args.log_every < args.rollout_steps:
            print(
                f"Step {step}/{args.total_steps} | "
                f"Policy Loss: {metrics['policy_loss']:.4f} | "
                f"Value Loss: {metrics['value_loss']:.4f} | "
                f"Entropy: {metrics['entropy']:.4f} | "
                f"Approx KL: {metrics['approx_kl']:.4f}"
            )
    
    print("\n" + "="*60)
    print("训练完成")
    print("="*60)
    
    # 保存智能体（可选）
    if args.save_agent:
        print(f"\n保存智能体到: {args.save_agent}")
        os.makedirs(os.path.dirname(args.save_agent) if os.path.dirname(args.save_agent) else '.', exist_ok=True)
        agent.save(args.save_agent)
        print("✓ 智能体保存成功")
    
    return 0


if __name__ == "__main__":
    exit(main())
