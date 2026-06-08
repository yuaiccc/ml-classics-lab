"""ML-Lab：CartPole A2C 训练过程导出（curves 家族，与 PPO 对照）。

运行：poetry run python runners/rl/cartpole_a2c.py
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
from tianshou.algorithm import A2C
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

OUTPUT = frames_dir() / "cartpole-a2c.json"


def main() -> None:
    task = "CartPole-v1"
    seed = 42
    hidden_sizes = [64, 64]
    lr = 7e-4
    gamma = 0.99
    gae_lambda = 0.95
    vf_coef = 0.5
    ent_coef = 0.01
    epoch = 50
    epoch_num_steps = 10000
    collection_step_num_env_steps = 2048
    update_step_num_repetitions = 1
    batch_size = 64
    num_training_envs = 16
    num_test_envs = 10
    buffer_size = 20000

    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make(task)
    training_envs = ts.env.DummyVectorEnv([lambda: gym.make(task) for _ in range(num_training_envs)])
    test_envs = ts.env.DummyVectorEnv([lambda: gym.make(task) for _ in range(num_test_envs)])

    state_shape = env.observation_space.shape
    action_shape = env.action_space.n

    net_a = Net(state_shape=state_shape, hidden_sizes=hidden_sizes, activation=nn.Tanh)
    actor = DiscreteActor(preprocess_net=net_a, action_shape=action_shape, softmax_output=False)
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
        actor=actor, dist_fn=dist_fn, action_space=env.action_space,
        action_scaling=False, action_bound_method=None,
    )
    optim = AdamOptimizerFactory(lr=lr)
    algorithm = A2C(
        policy=policy, critic=critic, optim=optim,
        gamma=gamma, gae_lambda=gae_lambda, vf_coef=vf_coef, ent_coef=ent_coef,
    )

    buffer = VectorReplayBuffer(buffer_size, num_training_envs)
    training_collector = Collector[CollectStats](algorithm, training_envs, buffer, exploration_noise=True)
    test_collector = Collector[CollectStats](algorithm, test_envs)

    def stop_fn(mean_rewards: float) -> bool:
        return mean_rewards >= env.spec.reward_threshold

    logger = EpochCaptureLogger()
    print("开始训练 A2C ...")
    result = algorithm.run_training(OnPolicyTrainerParams(
        training_collector=training_collector, test_collector=test_collector,
        max_epochs=epoch, epoch_num_steps=epoch_num_steps,
        collection_step_num_env_steps=collection_step_num_env_steps,
        update_step_num_repetitions=update_step_num_repetitions,
        test_step_num_episodes=num_test_envs, batch_size=batch_size,
        stop_fn=stop_fn, test_in_training=True, logger=logger,
    ))
    print(f"训练完成，最佳奖励 {result.best_reward:.1f}，捕获 {len(logger.records)} 帧")

    meta = Meta(
        id="cartpole-a2c", title="CartPole · A2C", algorithm="A2C", category="rl", source="python",
        abstract="A2C（优势演员-评论家）是最朴素的策略梯度+基线方法。在 CartPole 上训练奖励逐步上升，可与 PPO 对照——PPO 本质是带裁剪的更稳健的 A2C。",
        description="A2C 在 CartPole-v1 上的训练收敛过程，每个 epoch 记录测试平均奖励。",
        insight="A2C = 策略梯度 + 价值基线(critic)降低方差，同步更新。它简单但对步长敏感、易震荡；PPO 通过裁剪比率把它变得稳健，这是两者的核心差异。",
        hyperparams={"lr": lr, "gamma": gamma, "gae_lambda": gae_lambda, "ent_coef": ent_coef, "num_training_envs": num_training_envs},
    )
    export_trajectory(meta, logger, OUTPUT)
    env.close()


if __name__ == "__main__":
    main()
