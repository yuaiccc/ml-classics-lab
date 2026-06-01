"""
用奖励塑形（Reward Shaping）解决 MountainCar-v0

核心思路：原始 MountainCar 每步都是 -1，智能体永远体验不到"到达山顶"的正反馈。
我们给一个额外的塑形奖励：基于位置和速度，鼓励小车向右冲、利用动量荡上去。

这是 RL 中解决稀疏奖励问题的经典方法。
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


class MountainCarShaped(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        position, velocity = obs

        shaped = 0.0
        shaped += 100 * (position - self.env.unwrapped.state[0])
        shaped += 50 * abs(velocity)
        if position >= 0.5:
            shaped += 200

        reward = reward + shaped
        return obs, reward, terminated, truncated, info


def make_env():
    return MountainCarShaped(gym.make("MountainCar-v0"))


def main() -> None:
    seed = 42
    hidden_sizes = [128, 128, 128]
    lr = 1e-3
    gamma = 0.99
    n_step = 4
    target_update_freq = 500
    eps_train = 1.0
    eps_test = 0.01
    buffer_size = 100000
    epoch = 30
    epoch_num_steps = 10000
    collection_step_num_env_steps = 16
    batch_size = 128
    num_training_envs = 16
    num_test_envs = 10

    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_env()
    training_envs = ts.env.DummyVectorEnv(
        [lambda: make_env() for _ in range(num_training_envs)]
    )
    test_envs_raw = ts.env.DummyVectorEnv(
        [lambda: gym.make("MountainCar-v0") for _ in range(num_test_envs)]
    )

    space_info = SpaceInfo.from_env(env)
    state_shape = space_info.observation_info.obs_shape
    action_shape = space_info.action_info.action_shape
    print(f"环境: MountainCar-v0 (with Reward Shaping)")
    print(f"状态空间: {state_shape}, 动作空间: {action_shape}")

    net = Net(
        state_shape=state_shape,
        action_shape=action_shape,
        hidden_sizes=hidden_sizes,
    )
    optim = AdamOptimizerFactory(lr=lr)

    policy = DiscreteQLearningPolicy(
        model=net,
        action_space=gym.make("MountainCar-v0").action_space,
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
    test_collector = Collector[CollectStats](algorithm, test_envs_raw, exploration_noise=True)

    training_collector.reset()
    training_collector.collect(n_step=batch_size * num_training_envs)

    print(f"\n开始训练 DQN + Reward Shaping ...\n")

    def stop_fn(mean_rewards: float) -> bool:
        return mean_rewards >= -110

    def train_fn(epoch: int, env_step: int) -> None:
        eps = max(eps_train * (1 - 1e-5) ** env_step, eps_test)
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
    print(f"\n最终评估 (100 episodes, 原始奖励):")
    print(f"  平均奖励: {mean_reward:.1f} +/- {std_reward:.1f}")
    if mean_reward >= -110:
        print("  ✓ MountainCar 已被成功解决！")
    else:
        print("  ✗ 未达到目标")

    from frames_io import maybe_record

    # 录制用原始 MountainCar-v0（不带奖励塑形），保证回放是真实环境行为
    maybe_record(
        algorithm,
        "MountainCar-v0",
        meta={
            "id": "mountaincar-shaped",
            "title": "MountainCar · DQN + Reward Shaping（真实 rollout）",
            "family": "env",
            "algorithm": "DQN+Shaping",
            "envId": "MountainCar-v0",
            "description": "用势能奖励塑形破解稀疏奖励：训练时加 shaped reward，回放在原始环境上。",
            "insight": "观测 = [位置, 速度]。奖励塑形给“朝目标移动/积累动量”额外正反馈，让 DQN 也能学会上山。",
            "hyperparams": {"lr": "1e-3", "gamma": 0.99},
        },
    )


if __name__ == "__main__":
    main()
