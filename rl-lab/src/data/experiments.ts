export interface Experiment {
  id: string;
  env: string;
  algorithm: string;
  algorithmType: "on-policy" | "off-policy";
  status: "success" | "failed" | "partial";
  finalReward: number;
  targetReward: number;
  trainingTime: number;
  epochs: number;
  totalSteps: number;
  curve: { epoch: number; reward: number }[];
  hyperparams: Record<string, string | number>;
  mdp: {
    S: string;
    A: string;
    P: string;
    R: string;
    gamma: number;
  };
  abstract: string;
  description: string;
  algorithmInsight: string;
  actionType: "discrete" | "continuous";
}

export const experiments: Experiment[] = [
  {
    id: "cartpole-ppo",
    env: "CartPole-v1",
    algorithm: "PPO",
    algorithmType: "on-policy",
    status: "success",
    finalReward: 499.4,
    targetReward: 475,
    trainingTime: 14.76,
    epochs: 4,
    totalSteps: 8192,
    curve: [
      { epoch: 0, reward: 22 },
      { epoch: 1, reward: 173 },
      { epoch: 2, reward: 304 },
      { epoch: 3, reward: 444 },
      { epoch: 4, reward: 500 },
    ],
    hyperparams: {
      lr: "2.5e-4",
      gamma: 0.99,
      gae_lambda: 0.95,
      eps_clip: 0.2,
      vf_coef: 0.5,
      ent_coef: 0.01,
      batch_size: 64,
      hidden_sizes: "[64, 64]",
      num_envs: 8,
    },
    mdp: {
      S: "4维连续 [位置, 速度, 角度, 角速度]",
      A: "2离散 [向左, 向右]",
      P: "经典力学倒立摆方程",
      R: "每步+1，杆子倒下结束",
      gamma: 0.99,
    },
    abstract:
      "解决离散动作空间下的平衡控制问题：如何通过有限的离散动作（左推/右推）保持一个不稳定系统的平衡。这是强化学习最经典的入门问题，也是验证 on-policy 算法有效性的基准。PPO 在 4 轮训练内达到满分 500，证明了裁剪策略梯度在稠密奖励场景下的高效收敛能力。",
    description:
      "经典入门问题：通过左右推小车保持杆子不倒。PPO 仅用 4 轮训练即达到满分 500，展示了 on-policy 算法在稠密奖励问题上的高效性。",
    algorithmInsight:
      "PPO 通过裁剪策略更新比率 (clip ratio=0.2) 限制每步变化幅度，在保持训练效率的同时避免策略崩溃。GAE (λ=0.95) 精确估计优势函数，正交初始化加速收敛。",
    actionType: "discrete",
  },
  {
    id: "mountaincar-dqn",
    env: "MountainCar-v0",
    algorithm: "DQN",
    algorithmType: "off-policy",
    status: "failed",
    finalReward: -200,
    targetReward: -110,
    trainingTime: 45.2,
    epochs: 50,
    totalSteps: 500000,
    curve: Array.from({ length: 50 }, (_, i) => ({
      epoch: i + 1,
      reward: -200,
    })),
    hyperparams: {
      lr: "1e-3",
      gamma: 0.99,
      n_step: 4,
      target_update: 500,
      eps_train: 0.3,
      batch_size: 128,
      hidden_sizes: "[128, 128, 128]",
      buffer_size: 100000,
      num_envs: 16,
    },
    mdp: {
      S: "2维连续 [位置, 速度]",
      A: "3离散 [左推, 不推, 右推]",
      P: "欠驱动物理引擎",
      R: "每步-1，到达山顶结束",
      gamma: 0.99,
    },
    abstract:
      "暴露稀疏奖励问题的根本困境：当环境中每个动作都给出相同的惩罚（-1），而唯一的正信号（到达山顶）在随机策略下几乎不可能被触发时，智能体如何学习？DQN 的 ε-贪心探索不足以偶然到达山顶，导致 50 万步训练后奖励始终 -200。这个实验不是失败，而是对 RL 核心难题——探索——的直接演示。",
    description:
      "稀疏奖励的经典难题：小车必须先退后利用动量荡上山，但随机策略几乎不可能到达山顶，DQN 完全学不到有效信号。50 轮训练奖励始终 -200。",
    algorithmInsight:
      "DQN 失败的根源：每步都是 -1 的惩罚，智能体从未体验过'到达山顶'的正反馈。这揭示了 RL 的核心挑战——探索（Exploration）：如果连'好东西'都没见过，怎么知道去追求它？",
    actionType: "discrete",
  },
  {
    id: "mountaincar-ppo",
    env: "MountainCar-v0",
    algorithm: "PPO",
    algorithmType: "on-policy",
    status: "success",
    finalReward: -109.7,
    targetReward: -110,
    trainingTime: 198.0,
    epochs: 45,
    totalSteps: 258048,
    curve: [
      { epoch: 1, reward: -200 },
      { epoch: 5, reward: -200 },
      { epoch: 10, reward: -200 },
      { epoch: 13, reward: -196 },
      { epoch: 15, reward: -193 },
      { epoch: 19, reward: -173 },
      { epoch: 21, reward: -140 },
      { epoch: 23, reward: -126 },
      { epoch: 26, reward: -118 },
      { epoch: 31, reward: -113 },
      { epoch: 39, reward: -113 },
      { epoch: 43, reward: -111 },
      { epoch: 45, reward: -110 },
    ],
    hyperparams: {
      lr: "3e-4",
      gamma: 0.99,
      gae_lambda: 0.98,
      eps_clip: 0.2,
      vf_coef: 0.5,
      ent_coef: 0.05,
      batch_size: 64,
      hidden_sizes: "[64, 64]",
      num_envs: 16,
    },
    mdp: {
      S: "2维连续 [位置, 速度]",
      A: "3离散 [左推, 不推, 右推]",
      P: "欠驱动物理引擎",
      R: "每步-1，到达山顶结束",
      gamma: 0.99,
    },
    abstract:
      "解决稀疏奖励下的探索问题：PPO 的随机策略（高斯采样 + 熵正则化）比 DQN 的 ε-贪心有更强的探索能力，能在 16 个并行环境中偶然触发'到达山顶'的信号。一旦获得正反馈，策略梯度迅速学会'先退后冲'的非直觉策略。这证明了 on-policy 算法在探索困难问题上的优势——策略本身就在持续探索，而非依赖外部噪声。",
    description:
      "PPO 凭借更强的探索能力攻克了 DQN 失败的 MountainCar：随机策略 + 熵正则化让智能体偶然到达山顶，随后迅速学会先退后冲的策略。45 轮训练达到 -109.7。",
    algorithmInsight:
      "PPO 成功而 DQN 失败的关键区别：①随机策略天然探索（输出动作概率分布而非 Q 值）；②熵正则化 (ent_coef=0.05) 鼓励保持探索；③16 个并行环境增加偶然发现山顶的概率；④GAE (λ=0.98) 更重视长期回报。",
    actionType: "discrete",
  },
  {
    id: "pendulum-sac",
    env: "Pendulum-v1",
    algorithm: "SAC",
    algorithmType: "off-policy",
    status: "success",
    finalReward: -146.9,
    targetReward: -150,
    trainingTime: 82.24,
    epochs: 3,
    totalSteps: 15360,
    curve: [
      { epoch: 0, reward: -1480 },
      { epoch: 1, reward: -195 },
      { epoch: 2, reward: -152 },
      { epoch: 3, reward: -147 },
    ],
    hyperparams: {
      lr_actor: "3e-4",
      lr_critic: "3e-4",
      lr_alpha: "3e-4",
      gamma: 0.99,
      tau: 0.005,
      alpha: "auto (target=-1)",
      batch_size: 256,
      hidden_sizes: "[128, 128]",
      buffer_size: 100000,
      num_envs: 8,
    },
    mdp: {
      S: "3维连续 [cosθ, sinθ, 角速度]",
      A: "1维连续 力矩∈[-2,2]",
      P: "倒立摆旋转动力学",
      R: "-(θ² + 0.1·θ̇² + 0.001·a²)",
      gamma: 0.99,
    },
    abstract:
      "解决连续动作空间下的精确控制问题：当动作不再是有限选项（左/右），而是连续实数（力矩 -2.0 到 +2.0 之间的任意值）时，如何让智能体学会精确的力矩控制？SAC 通过高斯策略输出连续动作分布，自动温度调节平衡探索与利用，3 轮训练即从杆子乱转收敛到稳定竖直。这是机器人控制、自动驾驶等连续控制场景的核心方法。",
    description:
      "连续动作空间的倒立摆控制：杆子可 360° 旋转，需要施加连续力矩将其摆到竖直位置。SAC 用高斯分布采样动作，3 轮即从 -1480 收敛到 -147。",
    algorithmInsight:
      "SAC 的三大创新：①双 Critic 取 min 防止 Q 值过估计；②自动温度 α 平衡探索与利用（熵正则化）；③重参数化技巧让梯度可传回 Actor。连续动作无法枚举，SAC 直接输出高斯分布参数 (μ, σ) 并采样。",
    actionType: "continuous",
  },
];
