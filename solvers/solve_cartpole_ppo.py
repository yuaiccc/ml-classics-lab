"""
用 Tianshou 的 PPO 算法解决 CartPole-v1 平衡问题

问题描述：
  CartPole 是经典的强化学习入门环境。一根杆子立在小车上，
  智能体需要通过向左或向右推小车来保持杆子不倒。
  - 状态空间：4 维 (小车位置, 小车速度, 杆子角度, 杆子角速度)
  - 动作空间：2 维离散 (向左推, 向右推)
  - 奖励：每保持一步得 1 分，最高 500 分
  - 成功标准：平均奖励 >= 475

算法选择：
  PPO (Proximal Policy Optimization) 是当前最流行的策略梯度算法之一，
  通过裁剪策略更新比率来保证训练稳定性。
"""

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

import tianshou as ts
from tianshou.algorithm import PPO
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import Collector, CollectStats, VectorReplayBuffer
from tianshou.trainer import OnPolicyTrainerParams
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.discrete import DiscreteActor, DiscreteCritic


def main() -> None:
    # ==================== 超参数设置 ====================
    task = "CartPole-v1"
    seed = 42
    hidden_sizes = [64, 64]
    lr = 2.5e-4
    gamma = 0.99
    gae_lambda = 0.95
    eps_clip = 0.2
    vf_coef = 0.5
    ent_coef = 0.01
    max_grad_norm = 0.5
    epoch = 50
    epoch_num_steps = 10000
    collection_step_num_env_steps = 2048
    update_step_num_repetitions = 10
    batch_size = 64
    num_training_envs = 8
    num_test_envs = 10
    buffer_size = 20000

    np.random.seed(seed)
    torch.manual_seed(seed)

    # ==================== 创建环境 ====================
    env = gym.make(task)
    training_envs = ts.env.DummyVectorEnv(
        [lambda: gym.make(task) for _ in range(num_training_envs)]
    )
    test_envs = ts.env.DummyVectorEnv(
        [lambda: gym.make(task) for _ in range(num_test_envs)]
    )

    state_shape = env.observation_space.shape
    action_shape = env.action_space.n
    print(f"状态空间: {state_shape}, 动作空间: {action_shape}")

    # ==================== 构建网络 ====================
    # Actor 网络：状态 -> 动作概率分布
    net_a = Net(
        state_shape=state_shape,
        hidden_sizes=hidden_sizes,
        activation=nn.Tanh,
    )
    actor = DiscreteActor(
        preprocess_net=net_a,
        action_shape=action_shape,
        softmax_output=False,
    )

    # Critic 网络：状态 -> 状态价值 V(s)
    net_c = Net(
        state_shape=state_shape,
        hidden_sizes=hidden_sizes,
        activation=nn.Tanh,
    )
    critic = DiscreteCritic(preprocess_net=net_c)

    # 正交初始化（PPO 训练的最佳实践）
    actor_critic = ActorCritic(actor, critic)
    for m in actor_critic.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.zeros_(m.bias)

    # ==================== 创建策略和算法 ====================
    def dist_fn(logits: torch.Tensor) -> Categorical:
        return Categorical(logits=logits)

    policy = ProbabilisticActorPolicy(
        actor=actor,
        dist_fn=dist_fn,
        action_space=env.action_space,
        action_scaling=False,
        action_bound_method=None,
    )

    optim = AdamOptimizerFactory(lr=lr)

    algorithm = PPO(
        policy=policy,
        critic=critic,
        optim=optim,
        gamma=gamma,
        gae_lambda=gae_lambda,
        eps_clip=eps_clip,
        vf_coef=vf_coef,
        ent_coef=ent_coef,
        max_grad_norm=max_grad_norm,
        advantage_normalization=True,
        recompute_advantage=True,
    )

    # ==================== 创建数据收集器 ====================
    buffer = VectorReplayBuffer(buffer_size, num_training_envs)
    training_collector = Collector[CollectStats](
        algorithm, training_envs, buffer, exploration_noise=True
    )
    test_collector = Collector[CollectStats](algorithm, test_envs)

    # ==================== 训练 ====================
    print("\n开始训练 PPO ...")
    print(f"目标: 平均奖励 >= {env.spec.reward_threshold}\n")

    def stop_fn(mean_rewards: float) -> bool:
        return mean_rewards >= env.spec.reward_threshold

    result = algorithm.run_training(
        OnPolicyTrainerParams(
            training_collector=training_collector,
            test_collector=test_collector,
            max_epochs=epoch,
            epoch_num_steps=epoch_num_steps,
            collection_step_num_env_steps=collection_step_num_env_steps,
            update_step_num_repetitions=update_step_num_repetitions,
            test_step_num_episodes=num_test_envs,
            batch_size=batch_size,
            stop_fn=stop_fn,
            test_in_training=True,
        )
    )

    print(f"\n训练完成! 总耗时: {result.timing.total_time:.2f} 秒")
    print(f"最佳奖励: {result.best_reward:.1f}")

    # ==================== 最终评估 ====================
    test_collector.reset()
    eval_result = test_collector.collect(n_episode=100)
    mean_reward = eval_result.returns.mean()
    std_reward = eval_result.returns.std()
    print(f"\n最终评估 (100 episodes):")
    print(f"  平均奖励: {mean_reward:.1f} +/- {std_reward:.1f}")
    if mean_reward >= env.spec.reward_threshold:
        print("  ✓ 达到目标！CartPole 已被成功解决！")
    else:
        print("  ✗ 未达到目标，可尝试增加训练轮次或调整超参数")

    # ==================== 录制 rollout 供前端回放（M3）====================
    from frames_io import maybe_record

    maybe_record(
        algorithm,
        task,
        meta={
            "id": "cartpole-ppo",
            "title": "CartPole · PPO（真实 rollout）",
            "family": "env",
            "algorithm": "PPO",
            "envId": "CartPole-v1",
            "description": "训练好的 PPO 策略在 CartPole-v1 上的一条真实回放：靠左右推车保持杆子竖直。",
            "insight": "观测 = [小车位置, 小车速度, 杆角度, 杆角速度]。策略每步根据这 4 个量决定向左/向右推，把杆角度稳在 0 附近。撑满 500 步即满分。",
            "hyperparams": {"lr": "2.5e-4", "gamma": 0.99, "eps_clip": 0.2},
        },
    )

    env.close()


if __name__ == "__main__":
    main()
