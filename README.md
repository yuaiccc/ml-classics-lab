# ml-classics-lab

> 机器学习经典问题复现 Lab —— 我的学习笔记。
> 每个经典算法都亲手跑一遍，并配上**逐步收敛的过程动画**。

这个仓库分两部分：

- **`solvers/`** —— 用 [Tianshou](https://github.com/thu-ml/tianshou) 复现强化学习经典环境的脚本（每个脚本是一个独立可跑的实验 + 笔记）。
- **`rl-lab/`** —— 一个 React + Vite 前端「**算法过程动画实验台**」（主页即是）。左侧是一棵 **AI 发展脉络树**（分支=方法谱系，叶子=实验，带年代），31 个经典算法（含真实数据 Iris、评估与泛化、监督/无监督/深度学习/生成模型/强化学习）统一在一个时间轴播放器里边看边学。

完整的平台设计、统一帧契约、以及四大类（监督 / 无监督 / 深度学习 / 强化学习）的算法清单 checklist 见 [ML_LAB_DESIGN.md](ML_LAB_DESIGN.md)。
**维护 / 二次开发请先读 [HANDOFF.md](HANDOFF.md)**（架构、如何加新算法、已知的坑）。

---

## 🎓 新手怎么学（推荐路线）

不用先懂数学，**先看动画建立直觉**：

```bash
cd rl-lab && npm install && npm run dev   # 打开 http://localhost:5180
```

打开后就是**算法过程动画实验台**，左侧是学习路线，每个算法都有：

- **过程动画**：按 ▶️ 播放，看算法怎么一步步收敛；可拖进度条、单步慢看、调速、换数据
- **小白讲解**（动画下方）：解决什么问题 / 直觉理解 / 看动画注意什么 / 关键概念 / 动手试试

建议顺序：**线性回归 →（看懂梯度下降）→ 逻辑回归 / 感知机 / SVM →（看懂决策边界）→ 决策树 / KNN / AdaBoost →（看懂非线性/集成）→ K-Means / DBSCAN / PCA →（无监督）→ MLP / RNN / CNN / Self-Attention / Mamba →（深度学习时间线）→ GAN / Diffusion →（生成模型）→ Q-Learning / CartPole（强化学习）**。

`/lab` 共 **31 个 demo**（含真实数据 Iris、评估与泛化），按 AI 发展史排开——从 **1958 感知机** 一路到 **2017 Transformer、2020 Diffusion、2023 Mamba**，覆盖符号/统计学习、联想记忆、深度学习、生成模型、强化学习几乎所有里程碑。其中 8 个里程碑实验（SVM / AdaBoost / Hopfield / RNN / Word2Vec / GAN / Diffusion / Q-Learning）全部浏览器端手写、含真实训练、零 ML 库。

学有余力再回头看 `solvers/` 里的 Python 代码和 [ML_LAB_DESIGN.md](ML_LAB_DESIGN.md) 的算法清单。

---

## 已复现的强化学习实验

| 环境 | 算法 | 结果 | 笔记要点 |
|------|------|------|---------|
| CartPole-v1 | **PPO** (on-policy) | ✅ 4 轮达满分 500 | 稠密奖励下裁剪策略梯度的高效收敛 |
| MountainCar-v0 | **DQN** (off-policy) | ❌ 始终 -200 | 稀疏奖励 + ε-贪心探索不足，学不到信号 |
| MountainCar-v0 | **PPO** (on-policy) | ✅ 45 轮达 -109.7 | 随机策略 + 熵正则化攻克 DQN 失败的探索难题 |
| MountainCar-v0 | **DQN + Reward Shaping** | — | 奖励塑形破解稀疏奖励；训练用塑形环境、测试用原始环境 |
| Pendulum-v1 | **SAC** (off-policy) | ✅ 3 轮收敛到 -147 | 连续动作空间的精确控制，自动温度调节 |

> 一组有意思的对照：**MountainCar 同一个稀疏奖励难题，用了 DQN / PPO / DQN+塑形 三种打法**，直观对比"探索"这个 RL 核心挑战的不同解法。

---

## 快速开始

### 1. 复现 RL 实验（Python）

```bash
pip install -r requirements.txt
python solvers/solve_cartpole_ppo.py      # PPO 解 CartPole
python solvers/solve_mountaincar_ppo.py   # PPO 解 MountainCar
python solvers/solve_pendulum_sac.py      # SAC 解 Pendulum
# ...
```

### 2. 启动可视化 Lab（前端）

```bash
cd rl-lab
npm install
npm run dev        # http://localhost:5180
```

打开 http://localhost:5180/ 就是**算法过程动画实验台**：左侧学习路线选实验，右侧时间轴播放器可拖动、单步、调速、重新生成数据，每个实验下方配「小白教程 + 真实源码」。

---

## 路线图

参见 [ML_LAB_DESIGN.md §5 算法清单](ML_LAB_DESIGN.md) 与 §6 里程碑：

- [x] **M1** 帧契约 + 通用播放器（梯度下降 / K-Means 验证）
- [x] **M2** 监督/无监督浏览器端算法 —— 实验台 `/lab` 现有 **8 种**：
  - 回归：线性回归（梯度下降）
  - 分类：逻辑回归、感知机、决策树（CART/Gini）、KNN
  - 聚类：K-Means、DBSCAN
  - 降维：PCA（幂迭代）
- [x] **M3** RL 接入统一播放器：Python 录制管线 → JSON 帧 → 前端 `env` 家族回放
  - **5 个 RL 回放**：CartPole·PPO、MountainCar·DQN(失败)/PPO/Shaped、Pendulum·SAC
  - MountainCar 三连对照：DQN 失败 → PPO 换算法成功 → Shaped 改奖励成功
  - **经典控制对照**：CartPole 状态反馈控制器（浏览器端，不学习），演示"RL 之前怎么解"
- [x] **M4** 深度学习
  - [x] **MLP**（手写反向传播）—— 在同心圆数据上学非线性边界，对比逻辑回归
  - [x] **CNN**（PyTorch）—— 横线/竖线分类，卷积核随训练变成边缘检测器 + 激活图可视化
  - [x] **Self-Attention**（PyTorch）—— 序列反转任务，注意力矩阵收敛成反对角线 = BERT/GPT 的核心机制
  - [x] **Mamba / 选择性 SSM**（浏览器手写）—— 选择性扫描：input-dependent 门控决定记/忘，线性时间，注意力之后的前沿替代架构

### 重新生成 / 新增 RL 回放帧

每个 `solvers/solve_*.py` 都内置了录制开关。在 `tianshou` 环境里加环境变量运行即可，
训练结束后会把一条真实 episode 导出到 `rl-lab/src/data/frames/<id>.json` 供前端 `/lab` 回放：

```bash
conda activate tianshou
RECORD_FRAMES=1 python solvers/solve_pendulum_sac.py      # → pendulum-sac.json
RECORD_FRAMES=1 python solvers/solve_mountaincar_ppo.py   # → mountaincar-ppo.json
python solvers/train_cnn.py                               # → cnn-shapes.json（CNN，纯 PyTorch）
python solvers/train_transformer.py                       # → attention-reverse.json（自注意力）
```

前端 `env` 渲染器已支持 CartPole / MountainCar / Pendulum 三种环境，生成 JSON 后
在 `rl-lab/src/pages/AlgorithmLab.tsx` 的 `DEMOS` 里加一条 import 即可显示。

---

*基于 [Tianshou](https://github.com/thu-ml/tianshou) 深度强化学习库。本仓库为个人学习笔记，与上游 Tianshou 无关。*
