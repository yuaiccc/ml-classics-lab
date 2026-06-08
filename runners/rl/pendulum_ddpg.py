"""ML-Lab：Pendulum DDPG 训练过程导出（curves 家族，与 SAC/TD3 对照）。

运行：poetry run python runners/rl/pendulum_ddpg.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import tianshou as ts
from tianshou.algorithm import DDPG
from tianshou.algorithm.modelfree.ddpg import ContinuousDeterministicPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import Collector, CollectStats, VectorReplayBuffer
from tianshou.exploration import GaussianNoise
from tianshou.trainer import OffPolicyTrainerParams
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ContinuousActorDeterministic, ContinuousCritic
from tianshou.utils.space_info import SpaceInfo

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "runners" / "common"))
from trajectory import Meta  # noqa: E402
from capture_logger import EpochCaptureLogger, export_trajectory, frames_dir  # noqa: E402

OUTPUT = frames_dir() / "pendulum-ddpg.json"


def main() -> None:
    task = "Pendulum-v1"
    seed = 42
    hidden_sizes = [128, 128]
    lr = 3e-4
    gamma = 0.99
    tau = 0.005
    exploration_sigma = 0.1
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
    training_envs = ts.env.DummyVectorEnv([lambda: gym.make(task) for _ in range(num_training_envs)])
    test_envs = ts.env.DummyVectorEnv([lambda: gym.make(task) for _ in range(num_test_envs)])

    space_info = SpaceInfo.from_env(env)
    state_shape = space_info.observation_info.obs_shape
    action_shape = space_info.action_info.action_shape
    max_action = space_info.action_info.max_action

    net_a = Net(state_shape=state_shape, hidden_sizes=hidden_sizes)
    actor = ContinuousActorDeterministic(preprocess_net=net_a, action_shape=action_shape, max_action=max_action)
    net_c = Net(state_shape=state_shape, action_shape=action_shape, hidden_sizes=hidden_sizes, concat=True)
    critic = ContinuousCritic(preprocess_net=net_c)

    policy = ContinuousDeterministicPolicy(
        actor=actor, action_space=env.action_space,
        exploration_noise=GaussianNoise(sigma=exploration_sigma), action_scaling=True,
    )
    algorithm = DDPG(
        policy=policy, policy_optim=AdamOptimizerFactory(lr=lr),
        critic=critic, critic_optim=AdamOptimizerFactory(lr=lr), tau=tau, gamma=gamma,
    )

    buffer = VectorReplayBuffer(buffer_size, num_training_envs)
    training_collector = Collector[CollectStats](algorithm, training_envs, buffer, exploration_noise=True)
    test_collector = Collector[CollectStats](algorithm, test_envs)
    training_collector.reset()
    training_collector.collect(n_step=batch_size * num_training_envs)

    def stop_fn(mean_rewards: float) -> bool:
        return mean_rewards >= -150

    logger = EpochCaptureLogger()
    print("开始训练 DDPG ...")
    result = algorithm.run_training(OffPolicyTrainerParams(
        training_collector=training_collector, test_collector=test_collector,
        max_epochs=epoch, epoch_num_steps=epoch_num_steps,
        collection_step_num_env_steps=collection_step_num_env_steps,
        test_step_num_episodes=num_test_envs, batch_size=batch_size,
        update_step_num_gradient_steps_per_sample=1.0, stop_fn=stop_fn,
        test_in_training=True, logger=logger,
    ))
    print(f"训练完成，最佳奖励 {result.best_reward:.1f}，捕获 {len(logger.records)} 帧")

    meta = Meta(
        id="pendulum-ddpg", title="Pendulum · DDPG", algorithm="DDPG", category="rl", source="python",
        abstract="DDPG 把 DQN 的思想搬到连续动作：确定性策略 + Q 评论家 + 目标网络软更新，用高斯噪声做探索。在 Pendulum 上奖励逐步收敛，可与 TD3/SAC 对照。",
        description="DDPG 在 Pendulum-v1 上的训练收敛过程，每个 epoch 记录测试平均奖励。",
        insight="DDPG 是确定性策略梯度的代表：actor 直接输出动作、critic 估计 Q、靠外加高斯噪声探索。它样本高效但对超参敏感、易高估 Q——TD3 用双 Q+延迟更新修正，SAC 用最大熵进一步稳健化。",
        hyperparams={"lr": lr, "gamma": gamma, "tau": tau, "exploration_sigma": exploration_sigma, "num_training_envs": num_training_envs},
    )
    export_trajectory(meta, logger, OUTPUT)
    env.close()


if __name__ == "__main__":
    main()
