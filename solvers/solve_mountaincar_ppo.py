"""
用 PPO 算法解决 MountainCar-v0

核心难点：稀疏奖励——每步都是 -1，随机策略几乎不可能到达山顶。
PPO 的优势：随机策略天然有探索能力，配合较大的熵系数可以鼓励探索。
同时使用较大的初始收集步数，让策略有机会偶然到达山顶并获得正反馈。
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
    print(f"环境: {task}")
    print(f"状态空间: {state_shape}, 动作空间: {action_shape}")
    print(f"目标: 200步内到达山顶 (平均奖励 >= -110)")

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

    net_c = Net(
        state_shape=state_shape,
        hidden_sizes=hidden_sizes,
        activation=nn.Tanh,
    )
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

    print(f"\n开始训练 PPO on MountainCar ...\n")

    def stop_fn(mean_rewards: float) -> bool:
        return mean_rewards >= -110

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

    test_collector.reset()
    eval_result = test_collector.collect(n_episode=100)
    mean_reward = eval_result.returns.mean()
    std_reward = eval_result.returns.std()
    print(f"\n最终评估 (100 episodes):")
    print(f"  平均奖励: {mean_reward:.1f} +/- {std_reward:.1f}")
    if mean_reward >= -110:
        print("  ✓ MountainCar 已被成功解决！")
    else:
        print("  ✗ 未达到目标")

    from frames_io import maybe_record

    maybe_record(
        algorithm,
        task,
        meta={
            "id": "mountaincar-ppo",
            "title": "MountainCar · PPO（真实 rollout）",
            "family": "env",
            "algorithm": "PPO",
            "envId": "MountainCar-v0",
            "description": "PPO 攻克稀疏奖励的 MountainCar：学会“先退后冲”利用动量荡上山顶。",
            "insight": "观测 = [位置, 速度]。随机策略 + 熵正则化提供持续探索，偶然到达山顶后迅速学会非直觉的借力策略。",
            "hyperparams": {"lr": "3e-4", "gamma": 0.99, "ent_coef": 0.05},
        },
    )

    env.close()


if __name__ == "__main__":
    main()
