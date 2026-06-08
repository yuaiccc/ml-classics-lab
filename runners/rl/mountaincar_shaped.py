"""ML-Lab M3：MountainCar DQN（Reward Shaping 精简版）训练过程导出为 Trajectory JSON。

复用 solve_mountaincar_shaped.py 的配置（原脚本不改），导出
ml-lab/public/frames/mountaincar-dqn-shaped.json。测试在原始 MountainCar 上进行。

运行：poetry run python runners/rl/mountaincar_shaped.py
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

OUTPUT = frames_dir() / "mountaincar-dqn-shaped.json"


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
    # ==================== 超参数（同 solve_mountaincar_shaped.py）====================
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

    net = Net(state_shape=state_shape, action_shape=action_shape, hidden_sizes=hidden_sizes)
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

    print("开始训练 DQN + Reward Shaping（精简版，导出帧轨迹）...")
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
        id="mountaincar-dqn-shaped",
        title="MountainCar · DQN + Reward Shaping（精简）",
        algorithm="DQN",
        category="rl",
        source="python",
        abstract="奖励塑形的经典示范：给位置/速度额外引导信号，DQN 在 30 轮内学会借动量荡上山顶。",
        description="原始每步 -1 的稀疏奖励让 DQN 无从学习；塑形奖励鼓励向右冲、利用动量，快速收敛。",
        insight="奖励塑形是解决稀疏奖励的经典手段：把『遥远的成功』拆成『每步可感知的进步』。",
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

    env.close()


if __name__ == "__main__":
    main()
