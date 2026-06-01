# ML-Lab 设计文档

> 目标：把机器学习的经典算法**都做一遍**，每个都配**逐步收敛的过程动画**。
> 现状：已有 `rl-lab/`（React+Vite+TS，仅 RL）+ 5 个 `solve_*.py`（Tianshou）。
> 本文档定义如何把它演进成覆盖「监督 / 无监督 / 深度学习 / 强化学习」四大类的统一可视化平台。

---

## 1. 核心理念：统一的「帧轨迹」模型

四类算法表面差异很大，但"过程动画"的本质是一致的：**算法在迭代，每一步都有一个可画的状态**。
因此用一个统一数据契约把它们抹平 —— 一次实验就是一串 frames，前端只负责回放。

```
一次实验 run = Trajectory
Trajectory = {
  meta:    实验元信息（算法名、数据集、超参、目标）
  frames:  Frame[]        // 时间轴上的每一步
}
Frame = {
  iter:    number         // 第几步 / 第几个 epoch
  state:   <家族相关>      // 这一步的可画状态（见 §3）
  metrics: { [k]: number } // 标量指标 loss / reward / accuracy / inertia ...
}
```

> 设计要点：**先把 frame 契约和播放器定死**，后面 30+ 个算法都是往这个契约里填数据。
> 地基定错返工最贵，所以里程碑 1 只做地基验证（见 §6）。

---

## 2. 数据来源：浏览器原生 vs Python 预计算

两条产出路径，**输出同一种 frames 格式**，前端无感知差异。

| 方式 | 适用算法 | 好处 |
|------|---------|------|
| **浏览器原生 (TS)** | 梯度下降、线性/逻辑回归、感知机、K-Means、KNN、决策树、PCA | 真·交互：用户点击加数据点，实时重算重播 |
| **Python 预计算 → JSON** | SVM、随机森林/GBDT、t-SNE、所有神经网络、所有 RL | 训练太重，浏览器跑不动；导出帧后前端纯回放 |

简单算法亲手用 TS 实现更有教学价值，也支持交互式加点；重算法走 Python 管线。

---

## 3. Frame.state 的四种「家族可视化器」

每个算法归属一个可视化家族，前端为每个家族写一个 visualizer 组件，算法间复用。

### 3.1 `scatter-boundary`（监督学习 / 分类回归）
```ts
state = {
  points:   { x: number; y: number; label: number }[]   // 数据点（通常恒定）
  boundary: number[][]    // 决策边界：在网格上每个点的预测值/类别（用于画等高线/着色）
  // 或回归场景：
  fit?:     { slope: number; intercept: number } | number[]  // 当前拟合线/权重
}
```
动画：决策边界随迭代逐帧成形 / 拟合直线逐帧旋转贴合。

### 3.2 `clusters`（无监督 / 聚类）
```ts
state = {
  points:    { x: number; y: number; cluster: number }[]  // 点 + 当前归属
  centroids: { x: number; y: number }[]                   // 当前质心（K-Means/GMM）
  // 层次聚类用：
  merges?:   { a: number; b: number; height: number }[]   // 树状图合并记录
}
```
动画：质心逐帧移动、点的归属逐帧变色 / 树状图逐层生长。

### 3.3 `curves`（降维 / 训练曲线 / 通用标量）
```ts
state = {
  embedding?: { x: number; y: number; label: number }[]  // t-SNE/PCA 投影点
}
// metrics 时间序列单独画 loss/reward/accuracy 折线
```
动画：高维点逐帧展开到 2D / loss 曲线逐帧延伸。

### 3.4 `env`（强化学习）—— 复用现有组件
```ts
state = {
  observation: number[]   // env 状态（CartPole 的杆角度等）
  action?:     number
  reward?:     number
}
```
动画：env 行为动画（已有 `CartPoleViz` / `MountainCarViz` / `PendulumViz`）。

---

## 4. 目标目录结构

```
Tianshou/                         # 仓库根（Tianshou 本体不动）
├── ml-lab/                       # 由 rl-lab 升级/改名而来
│   ├── src/
│   │   ├── player/               # ★ 新核心：通用时间轴播放器
│   │   │   ├── TrajectoryPlayer.tsx   # 播放/暂停/进度条/调速
│   │   │   ├── useTrajectory.ts       # 加载 + 逐帧推进的 hook
│   │   │   └── types.ts               # Frame / Trajectory 契约（§1）
│   │   ├── visualizers/          # ★ 四种家族可视化器（§3）
│   │   │   ├── ScatterBoundary.tsx
│   │   │   ├── Clusters.tsx
│   │   │   ├── Curves.tsx
│   │   │   └── env/              # 现有 CartPoleViz 等迁移到这里
│   │   ├── algorithms/           # ★ 浏览器端 TS 实现
│   │   │   ├── gradient-descent.ts
│   │   │   ├── kmeans.ts
│   │   │   ├── logistic-regression.ts
│   │   │   └── ...               # 每个产出 Trajectory
│   │   ├── data/
│   │   │   └── experiments.ts    # 算法目录（扩展现有 schema）
│   │   ├── pages/                # Home / ExperimentDetail（已有）
│   │   └── ...
│   └── public/frames/            # 预计算 JSON（小样例可入库）
├── runners/                      # ★ Python 预计算管线
│   ├── common/
│   │   └── trajectory.py         # 写出符合 §1 契约的 JSON 的辅助函数
│   ├── supervised/               # sklearn：svm / random-forest / gbdt ...
│   ├── unsupervised/             # sklearn：tsne / dbscan / gmm ...
│   ├── deep/                     # PyTorch：mlp / cnn / transformer ...
│   └── rl/                       # 现有 solve_*.py 归到这里
├── frames/                       # 大体积预计算结果（.gitignore）
└── ML_LAB_DESIGN.md              # 本文档
```

> 迁移原则：**只做加法**。先不碰现有 5 个 `solve_*.py`，待里程碑 3 再接入播放器。

---

## 5. 算法清单（checklist）

图标含义：🌐 浏览器 TS 实现　🐍 Python 预计算　可视化家族见 §3

### 监督学习（supervised → `scatter-boundary`）
- [x] 🌐 线性回归 Linear Regression — 梯度下降拟合线逐帧贴合
- [x] 🌐 逻辑回归 Logistic Regression — 决策边界成形
- [x] 🌐 感知机 Perceptron — 每遇误分类点更新边界
- [x] 🌐 KNN — k 从小到大，展示偏差-方差权衡
- [x] 🌐 决策树 Decision Tree — 按深度逐层加深切分
- [ ] 🐍 SVM — 间隔/支持向量（SMO 迭代或仅终态）
- [ ] 🐍 朴素贝叶斯 Naive Bayes — 类条件分布
- [ ] 🐍 随机森林 Random Forest — 逐棵树加入
- [ ] 🐍 梯度提升 GBDT / XGBoost — 残差逐轮拟合

### 无监督学习（unsupervised → `clusters` / `curves`）
- [x] 🌐 K-Means — 质心逐帧移动
- [x] 🌐 PCA — 幂迭代求第一主成分，方向逐帧收敛
- [x] 🌐 DBSCAN — 密度可达扩张，簇逐个长出 + 噪声识别
- [ ] 🐍 层次聚类 Hierarchical — 树状图生长
- [ ] 🐍 高斯混合 GMM (EM) — 椭圆逐帧拟合
- [ ] 🐍 t-SNE / UMAP — 高维点逐帧展开

### 深度学习基础（deep → `scatter-boundary` / `curves`）
- [x] 🌐 梯度下降 Gradient Descent — 线性回归拟合线逐帧贴合
- [x] 🌐 MLP — 手写反向传播，同心圆非线性边界随 epoch 演化
- [ ] 🐍 MLP（更大规模）— 决策边界随 epoch 演化
- [x] 🐍 CNN — 卷积核 / 激活图可视化（横线vs竖线，`solvers/train_cnn.py`）
- [ ] 🐍 RNN / LSTM — 序列隐状态
- [ ] 🐍 Transformer — 注意力热图
- [ ] 🐍 Autoencoder — 潜空间 / 重建
- [ ] 🐍 GAN — 生成分布逐帧逼近

### 强化学习（rl → `env` + `curves`）
- [x] 🐍 DQN（MountainCar）— 已有 `solve_mountaincar_dqn.py`
- [x] 🐍 DQN + Reward Shaping — 已有 `solve_mountaincar_shaped.py`
- [x] 🐍 PPO（CartPole）— `solve_cartpole_ppo.py`，已生成 env 回放帧 ✅
- [x] 🐍 PPO（MountainCar）— `solve_mountaincar_ppo.py`，已生成 env 回放帧 ✅
- [x] 🐍 SAC（Pendulum）— `solve_pendulum_sac.py`，已生成 env 回放帧 ✅
- [ ] 🌐 Q-Learning / Sarsa — GridWorld 价值热图动画
- [ ] 🐍 A2C / DDPG / TD3 — 复用 Tianshou

---

## 6. 落地里程碑

| 里程碑 | 内容 | 产出 |
|--------|------|------|
| **M1 搭骨架** ✅ | 定义 frame schema + 通用播放器；用**梯度下降**+**K-Means**两个纯前端算法打通整链路 | 地基可用，动画跑通 |
| **M2 监督/无监督** ✅ | 浏览器端 8 种：线性/逻辑回归、感知机、决策树、KNN、K-Means、DBSCAN、PCA | 🌐 类基本铺满 |
| **M3 接入 RL** ✅ | Python 录制管线（`solvers/frames_io.py`）→ JSON 帧 → 前端 `env` 家族回放；CartPole·PPO 已生成验证，其余 4 脚本接好录制开关 | 新旧统一 |
| **M4 深度学习** | MLP → CNN → Transformer 注意力 | 四类齐全 |

> M3 录制管线：`RECORD_FRAMES=1 python solvers/solve_*.py`，训练后录一条 episode 导出
> `rl-lab/src/data/frames/<id>.json`（统一帧契约），前端按 `meta.envId` 选渲染器回放。

> M1 是关键：所有算法都依赖 frame 契约 + 播放器，先用两个最经典的过程动画验证它够不够通用。

---

## 7. 待确认的开放问题

1. **`rl-lab` 改名为 `ml-lab`**：是否接受重命名？（会改 import 路径，但只是前端目录）
2. **预计算帧体积**：深度/RL 的 frames 可能较大，是否用 `frames/` 目录 + `.gitignore`，只把小样例入库？
3. **是否引入 d3**：播放器的坐标轴/比例尺用 d3-scale 会省事，可否加这个依赖？
4. **交互优先级**：🌐 算法默认支持"用户加数据点重算"，是否所有都要做，还是先做 K-Means/感知机几个示范？
```
