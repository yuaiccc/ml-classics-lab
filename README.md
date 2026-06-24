# ml-classics-lab

> 一个「**算法过程动画 · 实验台**」——把 AI 发展史上的经典算法，每一个都亲手跑一遍，
> 配上**逐步收敛 / 可交互的过程动画**，边看边建立直觉。

打开就是一棵 **AI 发展脉络树**（分支 = 方法谱系，叶子 = 实验，带年代），
**53 个实验**统一在一个时间轴播放器里，从 **1958 感知机** 一路到
**Transformer 内部解剖、Mamba、MoE、RAG/Agent**，覆盖符号/统计学习、神经网络、
深度学习、生成模型、强化学习、大模型与 Agent 几乎所有里程碑。

仓库分两部分：

- **`rl-lab/`** —— React + Vite + TypeScript 前端实验台（主体）。统一「帧契约」+ 通用播放器 +
  各家族可视化器；浏览器端手写算法 + Python 预计算帧 + 本地大模型实时调用三种数据来源。
- **`solvers/`** —— 用 [Tianshou](https://github.com/thu-ml/tianshou) 复现强化学习经典环境的 Python 脚本
  （每个脚本独立可跑 + 录制过程帧供前端回放）。

> 平台设计 / 统一帧契约见 [ML_LAB_DESIGN.md](ML_LAB_DESIGN.md)；维护 / 加新算法见 [HANDOFF.md](HANDOFF.md)；
> 各板块的设计与实现计划见 [`docs/superpowers/`](docs/superpowers/)。

---

## 🚀 启动

```bash
cd rl-lab && npm install && npm run dev   # 打开 http://localhost:5180
```

左侧脉络树选实验；每个实验都有：

- **过程动画 / 交互**：按 ▶️ 播放看逐步收敛，或拖滑块 / 切变体实时对比；可单步、调速、换数据
- **小白教程**：解决什么问题 / 直觉理解 / 看动画注意什么 / 关键概念 / 动手试试
- **现实意义 + 参考文献**：每个实验解决的真实问题、真实应用，以及权威论文 / Wikipedia / Google ML 速成课链接
- **真实源码**：浏览器端手写算法可直接「查看源码」

```bash
npm test        # vitest：核心数学逻辑单测（47 个）
npm run build   # tsc + vite 生产构建
```

---

## 📚 实验全景（53 个，按 AI 发展史排开）

| 板块 | 实验 |
|------|------|
| **统计 · 符号学习** | 线性回归 · 逻辑回归 · 感知机(1958) · SVM(1995) · 决策树 · KNN · AdaBoost · K-Means · DBSCAN · PCA · Iris(Softmax/KNN, 真实数据) |
| **数据 · 评估 · 泛化** | 过拟合 · 正则化 L2 · ROC / 精确率-召回率（对齐 Google ML 速成课） |
| **联结主义 · 神经网络** | Hopfield(1982) · MLP(1986) · RNN(1997) · CNN(1998) · Word2Vec(2013) · YOLO + YOLO 版本演进(v1→v12) |
| **真实数据集 · 深度学习** | MNIST（CNN 卷积核 / 自编码器 / VAE / GAN）· IMDb（Tiny Transformer 注意力 / LSTM）· CIFAR-10 CNN |
| **🆕 LLM 架构解剖（Transformer 内部）** | **激活函数 · 注意力 KV 变体 · 位置编码 · 归一化 · 专家混合 MoE · 残差连接 · 线性·SSM · 稀疏·窗口注意力** |
| **大模型 · Agent 时代** | Self-Attention(2017) · Mamba/SSM(2023) · 本地 Qwen 语义嵌入 · RAG · Agentic RAG · ReAct Agent · DeerFlow 深度研究 |
| **生成模型** | GAN(2014) · Diffusion(2020) |
| **决策与控制（RL）** | CartPole 经典控制 · Q-Learning · CartPole/MountainCar/Pendulum 的 PPO/DQN/SAC + 奖励塑形 |

> 大模型 / Agent 部分（嵌入、RAG、ReAct、DeerFlow）**实时调用本地模型**（Ollama Qwen / 本地 DeerFlow），
> 不是录像回放。深度学习部分用真实数据集（MNIST/CIFAR/IMDb）的真实训练产物。

---

## 🧩 LLM 架构解剖（Transformer 内部）

新增板块，把现代大模型 Transformer block 的内部零件按「对比」逐类解剖，覆盖 35 篇论文。
每个实验内部用开关切换变体、滑块调参，核心数学都有 vitest 单测。

| 实验 | 对比变体 | 一眼看懂什么 |
|------|---------|-------------|
| 激活函数 | ReLU / GELU / SiLU / Swish / SwiGLU | 曲线+导数、β 滑块、门控 |
| 注意力 KV 变体 | MHA / MQA / GQA / MLA | Q→KV 共享合并、KV-cache 显存随变体缩小 |
| 位置编码 | 正弦 PE / RoPE / NoPE / YaRN | 向量旋转、相对位置不变性、YaRN 外推 |
| 归一化 | LayerNorm / RMSNorm / QKNorm / Pre·Post | 方向是否保持、残差流幅度随层深 |
| 专家混合 MoE | Dense / MoE / Switch / DeepSeekMoE | 路由点亮、稀疏激活、负载均衡 |
| 残差连接 | Plain / RC / HC / mHC / AttnResidual | 梯度沿层深消失 vs 被恒等捷径托住 |
| 线性·SSM | Lightning / DeltaNet / KDA / Mamba 1-3 | 定长状态 vs 注意力 KV-cache 随长度膨胀 |
| 稀疏·窗口注意力 | Full / Sparse / SWA / DSA / CSA | 掩码图案、FLOPs 占比 + **本地真实注意力切片** |

> 极新 / 未公开的变体（Mamba-3 / KDA / DSA / CSA 等）在界面里明确标注「按公开描述近似演示」。

### 🔬 形式化验证（Lean 4）

LLM 架构解剖板块中的 5 个核心组件已用 **Lean 4 + Mathlib4** 进行形式化验证，
在编译期数学证明其安全性质：

→ **[ml-lean-verify](https://github.com/yuaiccc/ml-lean-verify)**

| 组件 | 对应实验 | 证明的性质 |
|------|---------|-----------|
| Attention | 注意力 KV 变体 | softmax 分量非负（概率分布性质） |
| MoE | 专家混合 MoE | Top-k 稀疏激活占比上界 |
| Normalization | 归一化 | RMSNorm 纯缩放（方向不变），缩放因子非负 |
| Position Encoding | 位置编码 | RoPE 旋转角公式正确，YaRN = RoPE(base×scale) |
| Residual | 残差连接 | 所有残差变体梯度 ≥ 1（防止梯度消失） |

25 条定理 + 1200 轮差分测试全部通过。

---

## 🎮 强化学习实验（`solvers/`，Tianshou）

| 环境 | 算法 | 结果 | 笔记要点 |
|------|------|------|---------|
| CartPole-v1 | **PPO** | ✅ 4 轮达满分 500 | 稠密奖励下裁剪策略梯度的高效收敛 |
| MountainCar-v0 | **DQN** | ❌ 始终 -200 | 稀疏奖励 + ε-贪心探索不足，学不到信号 |
| MountainCar-v0 | **PPO** | ✅ 45 轮达 -109.7 | 随机策略 + 熵正则攻克 DQN 失败的探索难题 |
| MountainCar-v0 | **DQN + Reward Shaping** | — | 奖励塑形破解稀疏奖励；训练用塑形、测试用原始 |
| Pendulum-v1 | **SAC** | ✅ 3 轮收敛到 -147 | 连续动作的精确控制，自动温度调节 |

> 一组对照：**MountainCar 同一个稀疏奖励难题，用 DQN / PPO / DQN+塑形 三种打法**，
> 直观对比"探索"这个 RL 核心挑战的不同解法。

```bash
pip install -r requirements.txt
conda activate tianshou
RECORD_FRAMES=1 python solvers/solve_pendulum_sac.py   # 训练并导出过程帧 → 前端回放
```

---

## 🏗️ 架构一句话

所有算法的过程都归约到统一的 **帧轨迹契约** `Trajectory = { meta, frames[] }`，
通用播放器负责回放，每个实验绑定一个家族可视化器。加新算法 =
往 `DEMOS` 加一条（builder 产出 `Trajectory` + 指定 `Viz`）。详见 [HANDOFF.md](HANDOFF.md)。

---

*前端 React + Vite + TypeScript + Tailwind + recharts；RL 部分基于 [Tianshou](https://github.com/thu-ml/tianshou)。本仓库为个人学习项目，与上游 Tianshou 无关。*
