# ml-classics-lab

> 机器学习经典问题复现 Lab —— 我的学习笔记。
> 每个经典算法都亲手跑一遍，并配上**逐步收敛的过程动画**。

这个仓库分两部分：

- **`solvers/`** —— 用 [Tianshou](https://github.com/thu-ml/tianshou) 复现强化学习经典环境的脚本（每个脚本是一个独立可跑的实验 + 笔记）。
- **`rl-lab/`** —— 一个 React + Vite 前端，把实验结果做成可视化看板，并带一个「算法过程动画实验台」(`/lab`)，在浏览器里实时演示梯度下降、K-Means 等经典算法的收敛过程。

完整的平台设计、统一帧契约、以及四大类（监督 / 无监督 / 深度学习 / 强化学习）的算法清单 checklist 见 [ML_LAB_DESIGN.md](ML_LAB_DESIGN.md)。

---

## 🎓 新手怎么学（推荐路线）

不用先懂数学，**先看动画建立直觉**：

```bash
cd rl-lab && npm install && npm run dev   # 打开 http://localhost:5180
```

进首页点紫色「**算法过程动画 · 实验台**」按钮（或直接访问 `/lab`），里面每个算法都有：

- **过程动画**：按 ▶️ 播放，看算法怎么一步步收敛；可拖进度条、单步慢看、调速、换数据
- **小白讲解**（动画下方）：解决什么问题 / 直觉理解 / 看动画注意什么 / 关键概念 / 动手试试

建议顺序：**线性回归 →（看懂梯度下降）→ 逻辑回归 / 感知机 →（看懂决策边界）→ 决策树 / KNN →（看懂非线性）→ K-Means / DBSCAN →（无监督聚类）→ PCA →（降维）→ CartPole·PPO（强化学习）**。

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

- 首页：RL 实验看板（卡片式，点进去看每个实验的 MDP、训练曲线、env 动画）
- `/lab`：**算法过程动画实验台** —— 浏览器端实时计算，可拖时间轴、单步、调速、重新生成数据

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
  - 已生成并验证：**CartPole·PPO**（撑满 500 步）、**MountainCar·PPO**（先退后冲登顶）、**Pendulum·SAC**（摆杆立稳）
  - 其余脚本（DQN / Shaped）已接好录制开关，可自行生成（见下）
- [~] **M4** 深度学习（进行中）
  - [x] **MLP**（手写反向传播）—— 在同心圆数据上学非线性边界，对比逻辑回归
  - [ ] CNN（卷积核 / 激活图）、Transformer（注意力热图）—— 需 Python 预计算

### 重新生成 / 新增 RL 回放帧

每个 `solvers/solve_*.py` 都内置了录制开关。在 `tianshou` 环境里加环境变量运行即可，
训练结束后会把一条真实 episode 导出到 `rl-lab/src/data/frames/<id>.json` 供前端 `/lab` 回放：

```bash
conda activate tianshou
RECORD_FRAMES=1 python solvers/solve_pendulum_sac.py      # → pendulum-sac.json
RECORD_FRAMES=1 python solvers/solve_mountaincar_ppo.py   # → mountaincar-ppo.json
```

前端 `env` 渲染器已支持 CartPole / MountainCar / Pendulum 三种环境，生成 JSON 后
在 `rl-lab/src/pages/AlgorithmLab.tsx` 的 `DEMOS` 里加一条 import 即可显示。

---

*基于 [Tianshou](https://github.com/thu-ml/tianshou) 深度强化学习库。本仓库为个人学习笔记，与上游 Tianshou 无关。*
