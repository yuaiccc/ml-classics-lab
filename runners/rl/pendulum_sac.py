"""ML-Lab M3：Pendulum SAC 训练过程导出为 Trajectory JSON。

复用 solve_pendulum_sac.py 的配置（原脚本不改），导出
ml-lab/public/frames/pendulum-sac.json。

运行：poetry run python runners/rl/pendulum_sac.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import tianshou as ts
from tianshou.algorithm import SAC
from tianshou.algorithm.modelfree.sac import AutoAlpha, SACPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import Collector, CollectStats, VectorReplayBuffer
from tianshou.trainer import OffPolicyTrainerParams
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ContinuousActorProbabilistic, ContinuousCritic
from tianshou.utils.space_info import SpaceInfo

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "runners" / "common"))
from trajectory import Meta  # noqa: E402
from capture_logger import EpochCaptureLogger, export_trajectory, frames_dir  # noqa: E402

OUTPUT = frames_dir() / "pendulum-sac.json"


def main() -> None:
    # ==================== 超参数（同 solve_pendulum_sac.py）====================
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

    net_a = Net(state_shape=state_shape, hidden_sizes=hidden_sizes)
    actor = ContinuousActorProbabilistic(
        preprocess_net=net_a, action_shape=action_shape, unbounded=True
    )
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

    alpha = AutoAlpha(
        target_entropy=-action_dim,
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

    def stop_fn(mean_rewards: float) -> bool:
        return mean_rewards >= -150

    logger = EpochCaptureLogger()

    print("开始训练 SAC on Pendulum（导出帧轨迹）...")
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
            logger=logger,
        )
    )
    print(f"训练完成，最佳奖励 {result.best_reward:.1f}，捕获 {len(logger.records)} 帧")

    meta = Meta(
        id="pendulum-sac",
        title="Pendulum · SAC",
        algorithm="SAC",
        category="rl",
        source="python",
        abstract="连续控制 + 最大熵 RL：SAC 在 Pendulum 上把杆子摆到竖直，测试奖励从约 -1200 收敛到接近 0。",
        description="连续力矩控制，状态用 (cosθ, sinθ, θ̇) 编码。SAC 用双 Q + 自动温度 alpha 兼顾探索与利用。",
        insight="SAC 的最大熵目标鼓励保持策略随机性，在连续控制中兼顾探索与稳定，样本效率高。",
        hyperparams={
            "lr_actor": lr_actor,
            "lr_critic": lr_critic,
            "gamma": gamma,
            "tau": tau,
            "batch_size": batch_size,
            "num_training_envs": num_training_envs,
        },
    )
    export_trajectory(meta, logger, OUTPUT)

    env.close()


if __name__ == "__main__":
    main()
