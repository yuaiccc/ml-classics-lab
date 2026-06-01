# ml-classics-lab

> 机器学习经典问题复现 Lab —— 我的学习笔记。
> 每个经典算法都亲手跑一遍，并配上**逐步收敛的过程动画**。

这个仓库分两部分：

- **`solvers/`** —— 用 [Tianshou](https://github.com/thu-ml/tianshou) 复现强化学习经典环境的脚本（每个脚本是一个独立可跑的实验 + 笔记）。
- **`rl-lab/`** —— 一个 React + Vite 前端，把实验结果做成可视化看板，并带一个「算法过程动画实验台」(`/lab`)，在浏览器里实时演示梯度下降、K-Means 等经典算法的收敛过程。

完整的平台设计、统一帧契约、以及四大类（监督 / 无监督 / 深度学习 / 强化学习）的算法清单 checklist 见 [ML_LAB_DESIGN.md](ML_LAB_DESIGN.md)。

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
- [ ] **M2** 监督/无监督浏览器端算法（逻辑回归、感知机、决策树、KNN、PCA、DBSCAN）
- [ ] **M3** 把 RL 脚本接入统一播放器（Python 预计算帧）
- [ ] **M4** 深度学习（MLP → CNN → Transformer 注意力）

---

*基于 [Tianshou](https://github.com/thu-ml/tianshou) 深度强化学习库。本仓库为个人学习笔记，与上游 Tianshou 无关。*
