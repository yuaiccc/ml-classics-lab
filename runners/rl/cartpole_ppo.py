"""ML-Lab M3 竖切片：CartPole PPO 训练过程导出为 Trajectory JSON。

复用 solve_cartpole_ppo.py 的 env/PPO/网络配置（原脚本不改），
通过自定义 logger 捕获每个 epoch 的测试 reward，写出符合前端契约的
ml-lab/public/frames/cartpole-ppo.json。

运行：poetry run python runners/rl/cartpole_ppo.py
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

# 让脚本可直接运行时找到 runners/common 下的模块
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "runners" / "common"))
from trajectory import Meta  # noqa: E402
from capture_logger import EpochCaptureLogger, export_trajectory, frames_dir  # noqa: E402

OUTPUT = frames_dir() / "cartpole-ppo.json"


def main() -> None:
    # ==================== 超参数（同 solve_cartpole_ppo.py）====================
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
        return mean_rewards >= env.spec.reward_threshold

    logger = EpochCaptureLogger()

    print("开始训练 PPO（导出帧轨迹）...")
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

    # ==================== 写出 Trajectory ====================
    meta = Meta(
        id="cartpole-ppo",
        title="CartPole · PPO",
        algorithm="PPO",
        category="rl",
        source="python",
        abstract="PPO 在 CartPole-v1 上的训练收敛过程：测试平均奖励随 epoch 上升至满分。",
        description="经典入门问题，通过左右推小车保持杆子不倒。本轨迹回放每个训练 epoch 的测试平均奖励。",
        insight="PPO 通过裁剪策略更新比率限制每步变化幅度，GAE 精确估计优势函数，训练稳定快速收敛。",
        hyperparams={
            "lr": lr,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "eps_clip": eps_clip,
            "batch_size": batch_size,
            "num_training_envs": num_training_envs,
        },
    )
    export_trajectory(meta, logger, OUTPUT)

    env.close()


if __name__ == "__main__":
    main()
