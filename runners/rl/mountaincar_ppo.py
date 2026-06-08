"""ML-Lab M3：MountainCar PPO 训练过程导出为 Trajectory JSON。

复用 solve_mountaincar_ppo.py 的配置（原脚本不改），导出
ml-lab/public/frames/mountaincar-ppo.json。

运行：poetry run python runners/rl/mountaincar_ppo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

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

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "runners" / "common"))
from trajectory import Meta  # noqa: E402
from capture_logger import EpochCaptureLogger, export_trajectory, frames_dir  # noqa: E402

OUTPUT = frames_dir() / "mountaincar-ppo.json"


def main() -> None:
    # ==================== 超参数（同 solve_mountaincar_ppo.py）====================
    task = "MountainCar-v0"
    seed = 1
    hidden_sizes = [64, 64]
    lr = 3e-4
    gamma = 0.99
    gae_lambda = 0.98
    eps_clip = 0.2
    vf_coef = 0.5
    ent_coef = 0.05
    max_grad_norm = 0.5
    epoch = 200
    epoch_num_steps = 10000
    collection_step_num_env_steps = 2048
    update_step_num_repetitions = 10
    batch_size = 64
    num_training_envs = 16
    num_test_envs = 10
    buffer_size = 50000

    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make(task)
    training_envs = ts.env.DummyVectorEnv(
        [lambda: gym.make(task) for _ in range(num_training_envs)]
    )
    test_envs = ts.env.DummyVectorEnv(
        [lambda: gym.make(task) for _ in range(num_test_envs)]
    )

    state_shape = env.observation_space.shape
    action_shape = env.action_space.n

    net_a = Net(state_shape=state_shape, hidden_sizes=hidden_sizes, activation=nn.Tanh)
    actor = DiscreteActor(
        preprocess_net=net_a, action_shape=action_shape, softmax_output=False
    )
    net_c = Net(state_shape=state_shape, hidden_sizes=hidden_sizes, activation=nn.Tanh)
    critic = DiscreteCritic(preprocess_net=net_c)

    actor_critic = ActorCritic(actor, critic)
    for m in actor_critic.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.zeros_(m.bias)

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

    buffer = VectorReplayBuffer(buffer_size, num_training_envs)
    training_collector = Collector[CollectStats](
        algorithm, training_envs, buffer, exploration_noise=True
    )
    test_collector = Collector[CollectStats](algorithm, test_envs)

    def stop_fn(mean_rewards: float) -> bool:
        return mean_rewards >= -110

    logger = EpochCaptureLogger()

    print("开始训练 PPO on MountainCar（导出帧轨迹）...")
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
            logger=logger,
        )
    )
    print(f"训练完成，最佳奖励 {result.best_reward:.1f}，捕获 {len(logger.records)} 帧")

    meta = Meta(
        id="mountaincar-ppo",
        title="MountainCar · PPO",
        algorithm="PPO",
        category="rl",
        source="python",
        abstract="PPO 用随机策略的天然探索能力攻克 MountainCar 稀疏奖励，训练奖励从 -200 收敛到 >= -110。",
        description="稀疏奖励难题：每步 -1。PPO 的随机策略 + 较大熵系数提供探索，逐步学会借动量荡上山顶。",
        insight="对比 DQN：PPO 的随机策略天然探索，无需人工奖励塑形即可突破稀疏奖励困境。",
        hyperparams={
            "lr": lr,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "ent_coef": ent_coef,
            "batch_size": batch_size,
            "num_training_envs": num_training_envs,
        },
    )
    export_trajectory(meta, logger, OUTPUT)

    env.close()


if __name__ == "__main__":
    main()
