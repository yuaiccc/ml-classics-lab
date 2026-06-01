"""
用 Tianshou 的 SAC 算法解决 Pendulum-v1 问题

问题描述：
  一根杆子固定在一个轴上，可以自由旋转 360°。
  智能体需要在轴上施加力矩，把杆子从任意初始位置摆到竖直向上并保持。

  与 CartPole 的关键区别：
  - CartPole：杆子只能倒下（±24°），动作是离散的（左/右）
  - Pendulum：杆子可以转整圈，动作是连续的（力矩大小 ∈ [-2, 2]）

  - 状态空间：3 维 (cos(θ), sin(θ), θ̇)  — 角度用三角函数编码
  - 动作空间：1 维连续 (力矩 ∈ [-2, 2])
  - 奖励：-(θ² + 0.1·θ̇² + 0.001·action²)  — 越接近竖直、越稳、力越小越好
  - 成功标准：平均奖励接近 0（最好约 -120 左右）
"""

import gymnasium as gym
import numpy as np
import torch
from torch import nn

import tianshou as ts
from tianshou.algorithm import SAC
from tianshou.algorithm.modelfree.sac import AutoAlpha, SACPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import Collector, CollectStats, VectorReplayBuffer
from tianshou.trainer import OffPolicyTrainerParams
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ContinuousActorProbabilistic, ContinuousCritic
from tianshou.utils.space_info import SpaceInfo


def main() -> None:
    task = "Pendulum-v1"
    seed = 42
    hidden_sizes = [128, 128]
    lr_actor = 3e-4
    lr_critic = 3e-4
    lr_alpha = 3e-4
    gamma = 0.99
    tau = 0.005
    buffer_size = 100000
    epoch = 20
    epoch_num_steps = 5000
    collection_step_num_env_steps = 256
    batch_size = 256
    num_training_envs = 8
    num_test_envs = 10

    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make(task)
    training_envs = ts.env.DummyVectorEnv(
        [lambda: gym.make(task) for _ in range(num_training_envs)]
    )
    test_envs = ts.env.DummyVectorEnv(
        [lambda: gym.make(task) for _ in range(num_test_envs)]
    )

    space_info = SpaceInfo.from_env(env)
    state_shape = space_info.observation_info.obs_shape
    action_shape = space_info.action_info.action_shape
    action_dim = space_info.action_info.action_dim
    max_action = space_info.action_info.max_action

    print(f"环境: {task}")
    print(f"状态空间: {state_shape}  (cosθ, sinθ, 角速度)")
    print(f"动作空间: 连续 {action_shape}, 力矩 ∈ [{-max_action}, {max_action}]")
    print(f"目标: 杆子竖直向上并保持稳定 (奖励接近 0)")

    # Actor: 状态 → (均值μ, 标准差σ) → 高斯分布 → 采样 → tanh压缩 → 动作
    net_a = Net(
        state_shape=state_shape,
        hidden_sizes=hidden_sizes,
    )
    actor = ContinuousActorProbabilistic(
        preprocess_net=net_a,
        action_shape=action_shape,
        unbounded=True,
    )

    # Critic: (状态, 动作) → Q值
    # concat=True 让 Net 内部先拼接 obs 和 action，再送入 MLP
    net_c1 = Net(
        state_shape=state_shape,
        action_shape=action_shape,
        hidden_sizes=hidden_sizes,
        concat=True,
    )
    critic1 = ContinuousCritic(preprocess_net=net_c1)

    net_c2 = Net(
        state_shape=state_shape,
        action_shape=action_shape,
        hidden_sizes=hidden_sizes,
        concat=True,
    )
    critic2 = ContinuousCritic(preprocess_net=net_c2)

    # 自动调节温度参数 alpha（SAC 的熵正则化系数）
    target_entropy = -action_dim
    alpha = AutoAlpha(
        target_entropy=target_entropy,
        log_alpha=0.0,
        optim=AdamOptimizerFactory(lr=lr_alpha),
    )

    policy = SACPolicy(
        actor=actor,
        action_space=env.action_space,
        action_scaling=True,
        deterministic_eval=True,
    )

    algorithm = SAC(
        policy=policy,
        policy_optim=AdamOptimizerFactory(lr=lr_actor),
        critic=critic1,
        critic_optim=AdamOptimizerFactory(lr=lr_critic),
        critic2=critic2,
        critic2_optim=AdamOptimizerFactory(lr=lr_critic),
        tau=tau,
        gamma=gamma,
        alpha=alpha,
    )

    buffer = VectorReplayBuffer(buffer_size, num_training_envs)
    training_collector = Collector[CollectStats](
        algorithm, training_envs, buffer, exploration_noise=True
    )
    test_collector = Collector[CollectStats](algorithm, test_envs)

    training_collector.reset()
    training_collector.collect(n_step=batch_size * num_training_envs)

    print(f"\n开始训练 SAC ...")
    print(f"注意: 连续动作空间，SAC 用高斯分布采样 + tanh 压缩\n")

    def stop_fn(mean_rewards: float) -> bool:
        return mean_rewards >= -150

    result = algorithm.run_training(
        OffPolicyTrainerParams(
            training_collector=training_collector,
            test_collector=test_collector,
            max_epochs=epoch,
            epoch_num_steps=epoch_num_steps,
            collection_step_num_env_steps=collection_step_num_env_steps,
            test_step_num_episodes=num_test_envs,
            batch_size=batch_size,
            update_step_num_gradient_steps_per_sample=1.0,
            stop_fn=stop_fn,
            test_in_training=True,
        )
    )

    print(f"\n训练完成! 总耗时: {result.timing.total_time:.2f} 秒")
    print(f"最佳奖励: {result.best_reward:.1f}")

    test_collector.reset()
    eval_result = test_collector.collect(n_episode=100)
    mean_reward = eval_result.returns.mean()
    std_reward = eval_result.returns.std()
    print(f"\n最终评估 (100 episodes):")
    print(f"  平均奖励: {mean_reward:.1f} +/- {std_reward:.1f}")
    if mean_reward > -200:
        print("  ✓ Pendulum 控制效果良好！")
    else:
        print("  ✗ 效果一般，可尝试增加训练轮次")

    env.close()


if __name__ == "__main__":
    main()
