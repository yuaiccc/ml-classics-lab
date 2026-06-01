"""
用 Tianshou 的 DQN 算法解决 MountainCar-v0 问题

问题描述：
  一辆小车在两座山之间的谷底，目标是开到右侧山顶。
  但引擎力量不够，无法直接冲上去，必须先向左退再向右冲，
  利用动量荡上去——这就是"欠驱动"系统。

  - 状态空间：2 维 (小车位置, 小车速度)
  - 动作空间：3 维离散 (向左推, 不推, 向右推)
  - 奖励：每步 -1（鼓励尽快到达），到达山顶 +0 并结束
  - 成功标准：200 步内到达山顶

  这是 RL 中的经典难题：稀疏奖励 + 需要非直觉策略（先退后进）
"""

import gymnasium as gym
import numpy as np
import torch

import tianshou as ts
from tianshou.algorithm import DQN
from tianshou.algorithm.modelfree.dqn import DiscreteQLearningPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import Collector, CollectStats, VectorReplayBuffer
from tianshou.trainer import OffPolicyTrainerParams
from tianshou.utils.net.common import Net
from tianshou.utils.space_info import SpaceInfo


def main() -> None:
    task = "MountainCar-v0"
    seed = 42
    hidden_sizes = [128, 128, 128]
    lr = 1e-3
    gamma = 0.99
    n_step = 4
    target_update_freq = 500
    eps_train = 0.3
    eps_test = 0.01
    buffer_size = 100000
    epoch = 50
    epoch_num_steps = 10000
    collection_step_num_env_steps = 16
    batch_size = 128
    num_training_envs = 16
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
    print(f"环境: {task}")
    print(f"状态空间: {state_shape}  (位置 ∈ [-1.2, 0.6], 速度 ∈ [-0.07, 0.07])")
    print(f"动作空间: {action_shape}  (0=左推, 1=不推, 2=右推)")
    print(f"目标: 200 步内到达右侧山顶 (位置 >= 0.5)")

    net = Net(
        state_shape=state_shape,
        action_shape=action_shape,
        hidden_sizes=hidden_sizes,
    )
    optim = AdamOptimizerFactory(lr=lr)

    policy = DiscreteQLearningPolicy(
        model=net,
        action_space=env.action_space,
        eps_training=eps_train,
        eps_inference=eps_test,
    )

    algorithm = DQN(
        policy=policy,
        optim=optim,
        gamma=gamma,
        n_step_return_horizon=n_step,
        target_update_freq=target_update_freq,
    )

    buffer = VectorReplayBuffer(buffer_size, num_training_envs)
    training_collector = Collector[CollectStats](
        algorithm, training_envs, buffer, exploration_noise=True
    )
    test_collector = Collector[CollectStats](algorithm, test_envs, exploration_noise=True)

    training_collector.reset()
    training_collector.collect(n_step=batch_size * num_training_envs)

    print(f"\n开始训练 DQN ...")
    print(f"注意: MountainCar 是稀疏奖励问题，前期可能长时间没有正反馈\n")

    def stop_fn(mean_rewards: float) -> bool:
        return mean_rewards >= -110

    def train_fn(epoch: int, env_step: int) -> None:
        eps = max(eps_train * (1 - 5e-6) ** env_step, eps_test)
        policy.set_eps_training(eps)

    result = algorithm.run_training(
        OffPolicyTrainerParams(
            training_collector=training_collector,
            test_collector=test_collector,
            max_epochs=epoch,
            epoch_num_steps=epoch_num_steps,
            collection_step_num_env_steps=collection_step_num_env_steps,
            test_step_num_episodes=num_test_envs,
            batch_size=batch_size,
            update_step_num_gradient_steps_per_sample=0.0625,
            stop_fn=stop_fn,
            training_fn=train_fn,
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
    if mean_reward >= -110:
        print("  ✓ MountainCar 已被成功解决！")
    else:
        print("  ✗ 未达到目标，DQN 在稀疏奖励问题上需要更多探索")

    from frames_io import maybe_record

    maybe_record(
        algorithm,
        task,
        meta={
            "id": "mountaincar-dqn",
            "title": "MountainCar · DQN（真实 rollout）",
            "family": "env",
            "algorithm": "DQN",
            "envId": "MountainCar-v0",
            "description": "DQN 在稀疏奖励的 MountainCar 上的一条回放（通常学不到有效策略，可对比 PPO）。",
            "insight": "观测 = [位置, 速度]。稀疏奖励下 DQN 的 ε-贪心探索往往触发不了到达山顶的正信号。",
            "tutorial": {
                "problem": "DQN 在稀疏奖励的 MountainCar 上为什么会失败？这是探索难题的活教材。",
                "intuition": "每一步奖励都是 -1，唯一的正反馈在山顶。DQN 的 ε-贪心探索基本撞不到山顶——从没尝过“好结果”的味道，自然学不会去追求它。这一局就是它一直在山谷里来回晃，200 步都没上去。",
                "watch": [
                    "小车在谷底左右小幅晃动，始终冲不上去",
                    "Return 一路 -1 累加到 -200（满负），说明整局没到顶",
                    "这不是 bug，是“稀疏奖励 + 探索不足”的真实演示",
                ],
                "concepts": [
                    {"term": "稀疏奖励", "explain": "有用的正反馈极少出现，几乎学不到"},
                    {"term": "ε-贪心探索", "explain": "大概率走当前最优、小概率随机——撞不到罕见的山顶"},
                    {"term": "探索-利用困境", "explain": "没见过好结果，就不知道该去追求它"},
                ],
                "tryThis": "对照「MountainCar·PPO」（换算法成功）和「MountainCar·Shaped」（改奖励成功），看同一环境三种结局。",
            },
            "hyperparams": {"lr": "1e-3", "gamma": 0.99, "n_step": 4},
        },
    )

    env.close()


if __name__ == "__main__":
    main()
