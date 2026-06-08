"""ML-Lab M3：MountainCar DQN + Reward Shaping 训练过程导出为 Trajectory JSON。

复用 solve_mountaincar_dqn.py 的配置（原脚本不改），导出
ml-lab/public/frames/mountaincar-dqn.json。测试在原始 MountainCar 上进行，
因此记录的 reward 为未塑形的真实奖励（-200 ~ -110）。

运行：poetry run python runners/rl/mountaincar_dqn.py
"""

from __future__ import annotations

import sys
from pathlib import Path

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

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "runners" / "common"))
from trajectory import Meta  # noqa: E402
from capture_logger import EpochCaptureLogger, export_trajectory, frames_dir  # noqa: E402

OUTPUT = frames_dir() / "mountaincar-dqn.json"


class MountainCarShaped(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        position, velocity = obs
        shaped = 0.0
        shaped += 100 * (position - self.env.unwrapped.state[0])
        shaped += 50 * abs(velocity)
        if position >= 0.5:
            shaped += 200
        return obs, reward + shaped, terminated, truncated, info


def make_env():
    return MountainCarShaped(gym.make("MountainCar-v0"))


def main() -> None:
    # ==================== 超参数（同 solve_mountaincar_dqn.py）====================
    seed = 42
    hidden_sizes = [256, 256]
    lr = 5e-4
    gamma = 0.99
    n_step = 4
    target_update_freq = 320
    eps_train = 1.0
    eps_test = 0.01
    buffer_size = 200000
    epoch = 200
    epoch_num_steps = 10000
    collection_step_num_env_steps = 16
    batch_size = 256
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

    net = Net(state_shape=state_shape, action_shape=action_shape, hidden_sizes=hidden_sizes)
    optim = AdamOptimizerFactory(lr=lr)

    raw_env = gym.make("MountainCar-v0")
    policy = DiscreteQLearningPolicy(
        model=net,
        action_space=raw_env.action_space,
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
    test_collector = Collector[CollectStats](
        algorithm, test_envs_raw, exploration_noise=True
    )
    training_collector.reset()
    training_collector.collect(n_step=batch_size * num_training_envs)

    def stop_fn(mean_rewards: float) -> bool:
        return mean_rewards >= -110

    def train_fn(epoch: int, env_step: int) -> None:
        eps = max(eps_train * (1 - 1e-5) ** env_step, eps_test)
        policy.set_eps_training(eps)

    logger = EpochCaptureLogger()

    print("开始训练 DQN + Reward Shaping（导出帧轨迹）...")
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
            logger=logger,
        )
    )
    print(f"训练完成，最佳奖励 {result.best_reward:.1f}，捕获 {len(logger.records)} 帧")

    meta = Meta(
        id="mountaincar-dqn",
        title="MountainCar · DQN + Reward Shaping",
        algorithm="DQN",
        category="rl",
        source="python",
        abstract="DQN 配合奖励塑形与缓慢 ε 衰减攻克 MountainCar 稀疏奖励，测试奖励（原始）收敛到 >= -110。",
        description="DQN 本身因稀疏奖励难以学习，引入基于位置/速度的塑形奖励 + ε 从 1.0 缓慢衰减后成功到达山顶。",
        insight="DQN 需要人工奖励塑形来打破稀疏奖励困境——这与 PPO 的随机策略天然探索形成对比。",
        hyperparams={
            "lr": lr,
            "gamma": gamma,
            "n_step": n_step,
            "target_update_freq": target_update_freq,
            "batch_size": batch_size,
            "num_training_envs": num_training_envs,
        },
    )
    export_trajectory(meta, logger, OUTPUT)

    raw_env.close()


if __name__ == "__main__":
    main()
