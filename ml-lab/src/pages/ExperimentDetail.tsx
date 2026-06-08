import { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { experiments, getTrajectory } from "@/data/experiments";
import { loadTrajectory } from "@/player/loadTrajectory";
import type { Trajectory } from "@/player/types";
import { ArrowLeft, Clock, Zap, Layers, Target, Code2, GitBranch, Activity, BarChart3, CalendarDays, Info } from "lucide-react";
import CartPoleViz from "@/components/CartPoleViz";
import MountainCarViz from "@/components/MountainCarViz";
import PendulumViz from "@/components/PendulumViz";
import MnistViz from "@/components/MnistViz";
import MnistCnnViz from "@/components/MnistCnnViz";
import FrozenLakeViz from "@/components/FrozenLakeViz";
import ExperimentMotion from "@/components/ExperimentMotion";
import Cifar10Viz from "@/components/Cifar10Viz";
import ImdbSentimentViz from "@/components/ImdbSentimentViz";
import BreakoutViz from "@/components/BreakoutViz";
import TrajectoryPlayer from "@/player/TrajectoryPlayer";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  Bar,
  BarChart,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  Cell,
  Legend,
} from "recharts";

// 走 Python 预计算帧（public/frames/<id>.json）的实验 id
const PYTHON_FRAME_IDS = new Set<string>([
  "cartpole-ppo",
  "mountaincar-ppo",
  "mountaincar-dqn",
  "mountaincar-dqn-shaped",
  "pendulum-sac",
  "imdb-transformer",
  "mnist-cnn-kernels",
  "mnist-autoencoder",
  "imdb-lstm",
  "mnist-gan",
  "svm",
  "random-forest",
  "gbdt",
  "gmm",
  "tsne",
  "naive-bayes",
  "hierarchical",
  "mnist-vae",
  "cartpole-a2c",
  "pendulum-ddpg",
  "pendulum-td3",
]);

const CURVE_COLORS: Record<string, string> = {
  PPO: "#5b7b9a",
  DQN: "#7a8b5a",
  SAC: "#cc785c",
  "DQN + Reward Shaping": "#c99a4e",
  "Gradient Descent": "#cc785c",
  "K-Means": "#5b7b9a",
  "Linear Regression": "#b86a8a",
  "Logistic Regression": "#b86a8a",
  Perceptron: "#c99a4e",
  PCA: "#5b7b9a",
  MLP: "#b86a8a",
  "LeNet-style CNN": "#7a8b5a",
  "Small CNN": "#5b7b9a",
  "Q-Learning": "#5b7b9a",
  "TF-IDF + Logistic Regression": "#b86a8a",
  "DistilBERT fine-tune": "#5b7b9a",
  "Random Policy": "#908e85",
};

const STATUS_MAP: Record<string, { label: string; cls: string }> = {
  success: { label: "✓ 成功", cls: "status-success" },
  failed: { label: "✗ 失败", cls: "status-failed" },
  partial: { label: "△ 部分", cls: "status-partial" },
};

const CATEGORY_LABELS: Record<string, string> = {
  supervised: "监督学习",
  unsupervised: "无监督学习",
  deep: "深度学习",
  rl: "强化学习",
};

const MDP_ICONS = [
  { key: "S", label: "状态空间", emoji: "📊" },
  { key: "A", label: "动作空间", emoji: "🎮" },
  { key: "P", label: "转移函数", emoji: "🔄" },
  { key: "R", label: "奖励函数", emoji: "🏆" },
  { key: "gamma", label: "折扣因子", emoji: "⏳" },
];

const HYPERPARAM_HELP: Record<string, { description: string; recommendation: string }> = {
  K: {
    description: "聚类数量，决定 K-Means 要把数据分成几个簇。",
    recommendation: "相似入门数据常用 2-5；未知簇数可用 elbow/silhouette 先估计。",
  },
  C: {
    description: "逻辑回归的正则化强度倒数，越大约束越弱，模型更容易贴合训练数据。",
    recommendation: "文本 TF-IDF baseline 常从 0.5-4.0 网格搜索；IMDb 这里用 2.0。",
  },
  alpha: {
    description: "SAC 的熵温度系数，越大越鼓励探索；auto 表示自动调节。",
    recommendation: "连续控制推荐 auto；手动值常从 0.1-0.3 试起。",
  },
  batch_size: {
    description: "每次参数更新使用的样本数量，过小噪声大，过大更新更稳但更慢。",
    recommendation: "DQN 常用 64-256；PPO 常用 64-256；SAC 常用 256。",
  },
  dimensions: {
    description: "原始数据维度。",
    recommendation: "按数据本身决定；可视化教学常用 2D/3D。",
  },
  ent_coef: {
    description: "熵正则系数，越大越鼓励策略保持随机性，常用于增强探索。",
    recommendation: "CartPole 常用 0-0.01；MountainCar 这类稀疏奖励可试 0.02-0.1。",
  },
  eps_clip: {
    description: "PPO 裁剪阈值，限制新旧策略差异，防止一次更新走得太猛。",
    recommendation: "PPO 经典默认值 0.2；稳定后可在 0.1-0.3 之间微调。",
  },
  eps_train: {
    description: "训练阶段 epsilon-greedy 的随机动作概率，用来控制 DQN 探索强度。",
    recommendation: "简单任务可 0.1-0.3；稀疏奖励建议从 1.0 衰减到 0.01。",
  },
  gae_lambda: {
    description: "GAE 的偏差-方差折中参数，越接近 1 越看重长程回报估计。",
    recommendation: "PPO 常用 0.95；稀疏或长程任务可试 0.95-0.99。",
  },
  gamma: {
    description: "折扣因子，越接近 1 越重视未来奖励。",
    recommendation: "短 episode 常用 0.95-0.99；控制/RL 入门基准常用 0.99。",
  },
  init: {
    description: "K-Means 质心初始化方式，会影响最终收敛到哪个局部最优。",
    recommendation: "实际项目优先 k-means++；教学动画可用 random 展示局部最优风险。",
  },
  iterations: {
    description: "浏览器内算法迭代次数。",
    recommendation: "教学可视化常用 100-500；观察收敛曲线后再减少或增加。",
  },
  lr: {
    description: "学习率，控制每次参数更新的步长。",
    recommendation: "神经网络/RL 常从 1e-4 到 1e-3 试；线性小实验可用 1e-3 到 1e-1。",
  },
  lr_actor: {
    description: "SAC actor 网络学习率，控制策略网络更新步长。",
    recommendation: "SAC 连续控制常用 3e-4。",
  },
  lr_alpha: {
    description: "SAC 温度参数 alpha 的学习率。",
    recommendation: "通常与 actor/critic 一致，常用 3e-4。",
  },
  lr_critic: {
    description: "SAC critic 网络学习率，控制 Q 函数更新步长。",
    recommendation: "SAC 连续控制常用 3e-4；不稳定时先降到 1e-4。",
  },
  max_iters: {
    description: "K-Means 最大迭代次数。",
    recommendation: "小数据常用 30-100；若质心变化很小即可提前停止。",
  },
  max_features: {
    description: "TF-IDF 词表最多保留多少个 token/ngram 特征。",
    recommendation: "中等文本分类常用 10000-50000；数据越大可越高，但会增加内存和训练时间。",
  },
  max_iterations: {
    description: "算法最多允许的迭代次数。",
    recommendation: "感知机/梯度类教学实验常用 100-500。",
  },
  max_iter: {
    description: "优化器最多迭代次数，防止线性模型未收敛时过早停止。",
    recommendation: "Logistic Regression 常用 100-500；若出现收敛警告就增大。",
  },
  min_df: {
    description: "词至少出现在多少篇文档中才进入词表，用于过滤极罕见噪声词。",
    recommendation: "IMDb 这类 5 万样本文本常用 2-5；小数据可用 1-2。",
  },
  ngram_range: {
    description: "使用单词还是短语特征；1,2 表示同时使用 unigram 和 bigram。",
    recommendation: "情感分类常用 1,2；若数据很小可只用 1,1 降低过拟合。",
  },
  method: {
    description: "PCA 求主成分的数值方法。",
    recommendation: "小数据可直接特征分解；大数据常用 power iteration/SVD。",
  },
  model: {
    description: "使用的预训练模型或模型家族。",
    recommendation: "英文轻量文本分类可从 distilbert-base-uncased 起步；追求效果再换 BERT/RoBERTa。",
  },
  max_length: {
    description: "文本截断/填充到的最大 token 长度，越长能看更多上下文但计算更慢。",
    recommendation: "IMDb 常用 128-256；长评论任务可试 256-512。",
  },
  n_step: {
    description: "多步回报长度，让 DQN 用更远的未来奖励更新 Q 值。",
    recommendation: "DQN 常用 1-5；延迟奖励任务可试 3-5。",
  },
  num_envs: {
    description: "并行环境数量，越多采样越快，也能提高探索覆盖。",
    recommendation: "本机入门实验常用 8-16；CPU 足够时可到 32。",
  },
  optimizer: {
    description: "参数优化方法。",
    recommendation: "深度学习/RL 默认优先 Adam；凸小问题可用 SGD 展示原理。",
  },
  dropout: {
    description: "训练时随机屏蔽部分神经元，降低过拟合。",
    recommendation: "MLP/CNN 分类常用 0.1-0.5；MNIST MLP baseline 可从 0.2 试起。",
  },
  epochs: {
    description: "完整遍历训练集的次数。",
    recommendation: "MNIST 这类小数据 MLP 常用 5-20；若验证集不再提升就提前停止。",
  },
  episodes: {
    description: "评估或训练 episode 数量。",
    recommendation: "快速 smoke test 可用 5-20；正式 RL 评估常用 50-100 episode。",
  },
  frame_size: {
    description: "为了网页可视化导出的帧尺寸。",
    recommendation: "网页动画可用 42x42/84x84；DQN 训练经典预处理常用 84x84。",
  },
  frameskip: {
    description: "同一动作连续执行的帧数，用于降低 Atari 控制频率。",
    recommendation: "Atari DQN 常用 4。",
  },
  hidden_sizes: {
    description: "MLP 隐藏层宽度配置，决定模型容量。",
    recommendation: "MNIST MLP baseline 常用 128-512 宽度、1-3 层；过大需配合 dropout/正则化。",
  },
  conv_channels: {
    description: "每层卷积输出通道数，也就是会学习多少个卷积核/特征图。",
    recommendation: "MNIST 小 CNN 可用 8-32 通道；更复杂图像通常逐层增加到 64/128 以上。",
  },
  kernel_size: {
    description: "卷积核的空间尺寸，决定一次观察多大的局部区域。",
    recommendation: "MNIST/LeNet 常用 5×5；现代 CNN 更常见 3×3 堆叠。",
  },
  max_steps: {
    description: "单个 episode 最多执行步数，防止随机策略无限拖长。",
    recommendation: "Atari smoke test 可用 1000-5000；正式训练按环境默认或论文设定。",
  },
  slippery: {
    description: "FrozenLake 是否启用随机滑动转移，开启后同一动作不一定到达预期格子。",
    recommendation: "教学可先用 false 看懂策略，再用 true 观察随机性和鲁棒性。",
  },
  reward_shaping: {
    description: "额外奖励信号设计，用领域先验把稀疏奖励变得更密集。",
    recommendation: "MountainCar 常用位置、速度、到达目标 bonus；注意保持原始评估奖励。",
  },
  repeat_action_probability: {
    description: "ALE 的 sticky action 概率，控制动作是否可能被上一帧动作替代。",
    recommendation: "可复现 baseline 常设 0；更接近现代 Atari 评测可用 0.25。",
  },
  samples: {
    description: "生成或使用的数据点数量。",
    recommendation: "浏览器教学动画常用 60-200，既能看清结构也不卡顿。",
  },
  solver: {
    description: "Logistic Regression 的数值优化器。",
    recommendation: "稀疏 TF-IDF 二分类常用 liblinear 或 saga；多分类/大数据可试 saga/lbfgs。",
  },
  target_dims: {
    description: "PCA 降维后的目标维度。",
    recommendation: "可视化常用 2；压缩任务按累计解释方差选择，如 90%-95%。",
  },
  train_subset: {
    description: "为了快速复现实验而抽取的训练样本数量。",
    recommendation: "小样本 smoke test 可用 1000-5000；正式微调建议使用完整训练集。",
  },
  tau: {
    description: "SAC 目标网络软更新系数，越小目标网络变化越平滑。",
    recommendation: "SAC/TD3 常用 0.005；不稳定时可试 0.001-0.01。",
  },
};

export default function ExperimentDetail() {
  const { id } = useParams<{ id: string }>();
  const exp = experiments.find((e) => e.id === id);
  const hasPythonFrames = exp ? PYTHON_FRAME_IDS.has(exp.id) : false;
  const [remoteTrajectory, setRemoteTrajectory] = useState<Trajectory | null>(null);

  useEffect(() => {
    if (!exp || !hasPythonFrames) {
      setRemoteTrajectory(null);
      return;
    }
    let cancelled = false;
    loadTrajectory(exp.id)
      .then((t) => {
        if (!cancelled) setRemoteTrajectory(t);
      })
      .catch((err) => console.error(err));
    return () => {
      cancelled = true;
    };
  }, [exp, hasPythonFrames]);

  if (!exp) {
    return (
      <div className="max-w-5xl mx-auto px-4 py-20 text-center">
        <p className="text-stone-500">实验未找到</p>
        <Link to="/" className="text-[#cc785c] text-sm mt-2 inline-block">← 返回总览</Link>
      </div>
    );
  }

  const color = CURVE_COLORS[exp.algorithm] || "#cc785c";
  const status = STATUS_MAP[exp.status];
  const trajectory = exp.hasTrajectory ? getTrajectory(exp.id) : null;
  const metricLabel = exp.category === "rl" ? "奖励" : exp.category === "deep" ? "准确率" : "指标";
  const curveRewards = exp.curve.map((point) => point.reward);
  const curveBest = curveRewards.length > 0 ? Math.max(...curveRewards) : exp.finalReward;
  const curveWorst = curveRewards.length > 0 ? Math.min(...curveRewards) : exp.finalReward;
  const comparisonPool = experiments.filter((item) => item.category === "rl" && item.env === exp.env);
  const rewardComparison = (comparisonPool.length > 1 ? comparisonPool : experiments.filter((item) => item.category === "rl")).map((item) => ({
    id: item.id,
    label: `${item.env.replace("-v0", "").replace("-v1", "")}\n${item.algorithm.replace(" + Reward Shaping", "+Shaping")}`,
    algorithm: item.algorithm,
    finalReward: item.finalReward,
    targetReward: item.targetReward,
    status: item.status,
  }));

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <Link to="/" className="inline-flex items-center gap-1 text-sm text-stone-500 hover:text-[#cc785c] transition-colors mb-6">
        <ArrowLeft className="w-4 h-4" /> 返回总览
      </Link>

      <div className="mb-8">
        <div className="flex flex-wrap items-center gap-3 mb-3">
          <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-[rgba(91, 123, 154,0.1)] text-[#5b7b9a] border border-[rgba(91, 123, 154,0.2)]">
            {CATEGORY_LABELS[exp.category] || exp.category}
          </span>
          <span className={`text-xs font-mono px-2 py-0.5 rounded-full ${status.cls}`}>{status.label}</span>
        </div>
        <h1 className="font-serif text-4xl sm:text-5xl tracking-tight text-stone-900 mb-1">{exp.env}</h1>
        <div className="font-mono text-sm text-[#cc785c] mb-5">{exp.algorithm}</div>
        {exp.abstract && (
          <p className="text-lg text-stone-600 leading-relaxed max-w-2xl border-l-2 border-[#cc785c]/40 pl-4">
            {exp.abstract}
          </p>
        )}
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-8">
        <div className="glass rounded-lg p-3">
          <div className="flex items-center gap-2 text-xs text-stone-500 mb-1"><Zap className="w-3 h-3" />最终指标</div>
          <div className="text-lg font-bold font-mono" style={{ color }}>{exp.finalReward}</div>
        </div>
        <div className="glass rounded-lg p-3">
          <div className="flex items-center gap-2 text-xs text-stone-500 mb-1"><Target className="w-3 h-3" />目标</div>
          <div className="text-lg font-bold font-mono text-stone-700">{exp.targetReward}</div>
        </div>
        <div className="glass rounded-lg p-3">
          <div className="flex items-center gap-2 text-xs text-stone-500 mb-1"><Clock className="w-3 h-3" />耗时</div>
          <div className="text-lg font-bold font-mono text-stone-700">{exp.trainingTime > 0 ? `${exp.trainingTime}s` : "浏览器"}</div>
        </div>
        <div className="glass rounded-lg p-3">
          <div className="flex items-center gap-2 text-xs text-stone-500 mb-1"><Layers className="w-3 h-3" />迭代</div>
          <div className="text-lg font-bold font-mono text-stone-700">{exp.epochs}</div>
        </div>
      </div>

      {trajectory && <TrajectoryPlayer trajectory={trajectory} />}
      {remoteTrajectory && <TrajectoryPlayer trajectory={remoteTrajectory} />}

      {exp.id === "cartpole-ppo" && <CartPoleViz />}
      {(exp.id === "mountaincar-dqn" || exp.id === "mountaincar-dqn-shaped" || exp.id === "mountaincar-ppo") && (
        <MountainCarViz variant="shaped" />
      )}
      {exp.id === "pendulum-sac" && <PendulumViz />}
      {exp.id === "mnist-mlp" && <MnistViz />}
      {exp.id === "mnist-cnn" && <MnistCnnViz />}
      {exp.id === "frozenlake-qlearning" && <FrozenLakeViz />}
      {exp.id === "cifar10-cnn" && <Cifar10Viz />}
      {exp.id === "imdb-tfidf-logreg" && <ImdbSentimentViz />}
      {exp.id === "imdb-distilbert" && <ImdbSentimentViz resultPath="/results/imdb-distilbert.json" />}
      {exp.id === "breakout-random" && <BreakoutViz />}

      <ExperimentMotion experiment={exp} />

      {exp.curve.length > 0 && (
        <div className="glass rounded-xl p-6 mb-8">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-4">
            <h2 className="text-sm font-bold text-stone-600 flex items-center gap-2">
              <Activity className="w-4 h-4 text-[#5b7b9a]" /> 训练曲线
            </h2>
            <div className="grid grid-cols-3 gap-2 text-[10px] font-mono">
              <div className="rounded-md border border-stone-200 bg-stone-100/40 px-2 py-1">
                <div className="text-stone-500">BEST</div>
                <div className="text-stone-700">{curveBest.toFixed(1)}</div>
              </div>
              <div className="rounded-md border border-stone-200 bg-stone-100/40 px-2 py-1">
                <div className="text-stone-500">FINAL</div>
                <div className="text-stone-700">{exp.finalReward.toFixed(1)}</div>
              </div>
              <div className="rounded-md border border-stone-200 bg-stone-100/40 px-2 py-1">
                <div className="text-stone-500">RANGE</div>
                <div className="text-stone-700">{curveWorst.toFixed(0)}~{curveBest.toFixed(0)}</div>
              </div>
            </div>
          </div>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={exp.curve} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
                <defs>
                  <linearGradient id={`detail-grad-${exp.id}`} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={color} stopOpacity={0.3} />
                    <stop offset="100%" stopColor={color} stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(214, 210, 196,0.15)" />
                <XAxis dataKey="epoch" tick={{ fill: "#908e85", fontSize: 11, fontFamily: "JetBrains Mono" }} axisLine={{ stroke: "rgba(214, 210, 196,0.2)" }} />
                <YAxis tick={{ fill: "#908e85", fontSize: 11, fontFamily: "JetBrains Mono" }} axisLine={{ stroke: "rgba(214, 210, 196,0.2)" }} />
                <Tooltip contentStyle={{ background: "rgba(255, 255, 255,0.9)", border: "1px solid rgba(204, 120, 92,0.2)", borderRadius: "8px", fontSize: "12px", fontFamily: "JetBrains Mono" }} labelStyle={{ color: "#908e85" }} itemStyle={{ color }} formatter={(value: number) => [exp.category === "deep" ? `${(value * 100).toFixed(2)}%` : value.toFixed(1), metricLabel]} labelFormatter={(label) => `Epoch ${label}`} />
                <ReferenceLine y={exp.targetReward} stroke="rgba(201, 154, 78,0.5)" strokeDasharray="5 5" label={{ value: `目标 ${exp.targetReward}`, fill: "#c99a4e", fontSize: 10, fontFamily: "JetBrains Mono" }} />
                <Area type="monotone" dataKey="reward" stroke={color} strokeWidth={2} fill={`url(#detail-grad-${exp.id})`} dot={{ r: 3, fill: color, stroke: "#f0eee6", strokeWidth: 2 }} activeDot={{ r: 5, fill: color, stroke: "#f0eee6", strokeWidth: 2 }} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {exp.category === "rl" && (
        <div className="glass rounded-xl p-6 mb-8">
          <div className="flex items-center justify-between gap-3 mb-4">
            <h2 className="text-sm font-bold text-stone-600 flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-[#c99a4e]" /> Reward 对比
            </h2>
            <span className="text-[10px] text-stone-500 font-mono">
              {comparisonPool.length > 1 ? `${exp.env} 同环境` : "全部 RL 实验"}
            </span>
          </div>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={rewardComparison} margin={{ top: 8, right: 18, bottom: 42, left: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(214, 210, 196,0.15)" />
                <XAxis
                  dataKey="label"
                  interval={0}
                  tick={{ fill: "#908e85", fontSize: 10, fontFamily: "JetBrains Mono" }}
                  tickFormatter={(label: string) => label.split("\n").join(" ")}
                  axisLine={{ stroke: "rgba(214, 210, 196,0.2)" }}
                />
                <YAxis tick={{ fill: "#908e85", fontSize: 11, fontFamily: "JetBrains Mono" }} axisLine={{ stroke: "rgba(214, 210, 196,0.2)" }} />
                <Tooltip
                  contentStyle={{ background: "rgba(255, 255, 255,0.92)", border: "1px solid rgba(201, 154, 78,0.2)", borderRadius: "8px", fontSize: "12px", fontFamily: "JetBrains Mono" }}
                  labelStyle={{ color: "#908e85" }}
                  formatter={(value: number, name: string) => [value.toFixed(1), name === "finalReward" ? "最终 reward" : "目标 reward"]}
                />
                <Legend wrapperStyle={{ fontSize: 11, fontFamily: "JetBrains Mono", color: "#908e85" }} />
                <Bar dataKey="targetReward" name="目标 reward" radius={[4, 4, 0, 0]} fill="rgba(107, 106, 101,0.45)" />
                <Bar dataKey="finalReward" name="最终 reward" radius={[4, 4, 0, 0]}>
                  {rewardComparison.map((item) => (
                    <Cell key={item.id} fill={CURVE_COLORS[item.algorithm] || "#cc785c"} fillOpacity={item.status === "failed" ? 0.55 : 0.9} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <div className="glass rounded-xl p-6">
          <h2 className="text-sm font-bold text-stone-600 mb-4">⚙️ 超参数</h2>
          <div className="space-y-2">
            {Object.entries(exp.hyperparams).map(([key, val]) => (
              <div key={key} className="flex items-center justify-between py-1.5 border-b border-stone-200/50 last:border-0">
                <span className="flex items-center gap-1.5 text-xs text-stone-500 font-mono">
                  {key}
                  {HYPERPARAM_HELP[key] && (
                    <span className="relative inline-flex group">
                      <Info
                        className="w-3 h-3 text-stone-500 hover:text-[#5b7b9a] focus-visible:text-[#5b7b9a] transition-colors outline-none"
                        tabIndex={0}
                        role="img"
                        aria-label={`${key}: ${HYPERPARAM_HELP[key].description} 推荐值：${HYPERPARAM_HELP[key].recommendation}`}
                      />
                      <span className="pointer-events-none absolute left-1/2 top-5 z-20 hidden w-72 -translate-x-1/2 rounded-md border border-[#5b7b9a]/20 bg-stone-100 px-3 py-2 text-[11px] leading-relaxed text-stone-700 shadow-xl shadow-black/30 group-hover:block group-focus-within:block">
                        <span className="block text-stone-700">{HYPERPARAM_HELP[key].description}</span>
                        <span className="mt-2 block border-t border-stone-200 pt-2 text-[#5b7b9a]">
                          推荐：{HYPERPARAM_HELP[key].recommendation}
                        </span>
                      </span>
                    </span>
                  )}
                </span>
                <span className="text-xs text-stone-700 font-mono font-medium">{String(val)}</span>
              </div>
            ))}
          </div>
        </div>
        <div className="glass rounded-xl p-6">
          <h2 className="text-sm font-bold text-stone-600 mb-4">🧩 MDP 五元组</h2>
          <div className="space-y-3">
            {MDP_ICONS.map(({ key, label, emoji }) => {
              const value = key === "gamma" ? String(exp.mdp.gamma) : exp.mdp[key as keyof typeof exp.mdp];
              return (
                <div key={key} className="flex items-start gap-3">
                  <div className="w-8 h-8 rounded-lg bg-[rgba(204, 120, 92,0.1)] flex items-center justify-center text-sm shrink-0">{emoji}</div>
                  <div>
                    <div className="text-[10px] text-stone-500 font-mono uppercase">{label} ({key === "gamma" ? "γ" : key})</div>
                    <div className="text-xs text-stone-700">{value}</div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {exp.tianshou && (
        <div className="glass rounded-xl p-6 mb-8">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 mb-5">
            <h2 className="text-sm font-bold text-stone-600 flex items-center gap-2">
              <Code2 className="w-4 h-4 text-[#cc785c]" /> Tianshou 调用栈
            </h2>
            <span className="text-[10px] font-mono text-stone-500 border border-stone-200 rounded-md px-2 py-1">
              {exp.tianshou.sourceFile}
            </span>
          </div>

          <div className="mb-5">
            <div className="flex items-center gap-2 text-[10px] text-stone-500 font-mono uppercase mb-2">
              <GitBranch className="w-3 h-3" /> train flow
            </div>
            <div className="flex flex-wrap gap-2">
              {exp.tianshou.flow.map((step, index) => (
                <div key={`${step}-${index}`} className="flex items-center gap-2">
                  <span className="rounded-md border border-[#cc785c]/20 bg-[#cc785c]/5 px-2 py-1 text-[10px] text-stone-700 font-mono">
                    {step}
                  </span>
                  {index < exp.tianshou!.flow.length - 1 && <span className="text-stone-400 text-xs">→</span>}
                </div>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {exp.tianshou.modules.map((mod) => (
              <div key={mod.name} className="rounded-lg border border-stone-200 bg-stone-100/30 p-3">
                <div className="text-xs font-mono text-[#5b7b9a] mb-1">{mod.name}</div>
                <div className="text-xs text-stone-500 leading-relaxed">{mod.role}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="glass rounded-xl p-6 mb-8">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 mb-4">
          <h2 className="text-sm font-bold text-stone-600 flex items-center gap-2">
            <CalendarDays className="w-4 h-4 text-[#c99a4e]" /> 科学史节点
          </h2>
          <Link to="/timeline" className="text-[10px] font-mono text-[#c99a4e] hover:text-[#cc785c] transition-colors">
            查看完整时间轴 →
          </Link>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-[1fr_auto_1fr] gap-4 items-stretch">
          <div className="rounded-lg border border-stone-200 bg-stone-100/30 p-4">
            <div className="text-2xl font-bold font-mono text-[#5b7b9a] mb-1">{exp.history.problemYear}</div>
            <div className="text-[10px] text-stone-500 font-mono mb-2">PROBLEM PROPOSED</div>
            <div className="text-xs text-stone-700 leading-relaxed">{exp.history.problemLabel}</div>
          </div>
          <div className="hidden md:flex items-center text-stone-400 font-mono text-xs">→</div>
          <div className="rounded-lg border border-stone-200 bg-stone-100/30 p-4">
            <div className="text-2xl font-bold font-mono text-[#c99a4e] mb-1">{exp.history.breakthroughYear}</div>
            <div className="text-[10px] text-stone-500 font-mono mb-2">REPRESENTATIVE BREAKTHROUGH</div>
            <div className="text-xs text-stone-700 leading-relaxed">{exp.history.breakthroughLabel}</div>
          </div>
        </div>
        <p className="text-xs text-stone-500 leading-relaxed mt-4">{exp.history.note}</p>
      </div>

      <div className="glass rounded-xl p-6">
        <h2 className="text-sm font-bold text-stone-600 mb-3">💡 算法原理</h2>
        <p className="text-sm text-stone-600 leading-relaxed">{exp.algorithmInsight}</p>
      </div>
    </div>
  );
}
