# HANDOFF —— 维护者交接文档

> 给接手这个项目的人。读完这一篇，你应该能：跑起来、看懂架构、自己加一个新算法。
> 配套阅读：[README.md](README.md)（用户视角）、[ML_LAB_DESIGN.md](ML_LAB_DESIGN.md)（设计与算法清单）。

---

## 1. 这是什么

一个「机器学习经典算法复现 + 过程动画可视化」的学习 lab。两部分：

- **`solvers/`** —— Python 脚本，训练真实模型（RL 用 Tianshou，CNN/Transformer 用 PyTorch），把训练过程导出成 JSON 帧。
- **`rl-lab/`** —— React + Vite + TS 前端。核心是 `/lab` 路由（`AlgorithmLab.tsx`）：一个**统一的时间轴播放器**，回放任何算法的「过程动画」，每个算法配「小白教程 + 源码/真实训练」。

> 命名历史：前端目录仍叫 `rl-lab/`、首页仍叫「Tianshou RL Lab」，是早期只做 RL 时的遗留。现在已扩展到监督/无监督/深度学习/强化学习四大类。改名非必须，但若重构可考虑。

---

## 2. 跑起来

```bash
# 前端（开发）
cd rl-lab && npm install && npm run dev      # → http://localhost:5180（端口写死，strictPort）

# 类型检查 + 生产构建（改完务必跑）
cd rl-lab && npx tsc --noEmit && npx vite build

# Python 侧（生成帧），用 conda 环境 tianshou
conda activate tianshou
RECORD_FRAMES=1 python solvers/solve_cartpole_ppo.py   # RL：env 变量开关
python solvers/train_cnn.py                            # CNN
python solvers/train_transformer.py                    # 自注意力
```

**conda 环境 `tianshou`**：torch 2.12 / tianshou 2.0.1 / gymnasium 1.3.0 / numpy。
⚠️ **没有 torchvision、没有 sklearn** —— 所以 CNN/Transformer 用的是 numpy 合成数据，不要写依赖它们的代码。

---

## 3. 核心架构：统一「帧轨迹」契约

一切围绕这个数据结构（定义在 [`rl-lab/src/player/types.ts`](rl-lab/src/player/types.ts)）：

```ts
Trajectory = { meta: TrajectoryMeta, frames: Frame[] }
Frame      = { iter: number, state: <家族相关>, metrics?: {[k]: number} }
```

- **播放器**（[`player/useTrajectory.ts`](rl-lab/src/player/useTrajectory.ts) + [`TrajectoryPlayer.tsx`](rl-lab/src/player/TrajectoryPlayer.tsx)）：接收一个 Trajectory，提供播放/暂停/单步/拖拽/调速。**与具体算法完全解耦**。
- **可视化家族**：`meta.family` 标记数据形状，每个家族一个 State 类型 + 一个可视化器组件。

| family | State 类型 | 可视化器 | 用于 |
|--------|-----------|---------|------|
| `scatter-boundary` | `RegressionState` / `BoundaryState` | `RegressionPlot` / `BoundaryPlot` | 回归线 / 分类决策边界 |
| `clusters` | `ClusterState` | `ClustersPlot` | K-Means / DBSCAN |
| `pca` | `PCAState` | `PCAPlot` | PCA 主成分 |
| `env` | `EnvState` | `EnvPlot`（按 `meta.envId` 切 CartPole/MountainCar/Pendulum） | RL 回放 |
| `cnn` | `CnnState` | `CnnPlot` | 卷积核 + 激活图 |
| `attention` | `AttentionState` | `AttentionPlot` | 自注意力热图 |
| `ssm` | `SSMState` | `SsmPlot` | Mamba 选择性扫描（浏览器端） |
| `gridworld` | `GridWorldState` | `GridWorldPlot` | 表格 Q-Learning 价值热图 |
| `hopfield` | `HopfieldState` | `HopfieldPlot` | Hopfield 联想记忆图案 |
| `rnn` | `RnnState` | `RnnPlot` | RNN 隐藏状态轨迹 |
| `embedding` | `EmbeddingState` | `Word2VecPlot` | 词向量 2D 散点 |

> GAN / Diffusion 复用 `clusters` 家族的 `ClustersPlot`（真/假、生成/目标用 cluster 区分）；
> SVM / AdaBoost 复用 `scatter-boundary` 的 `BoundaryPlot`。小型 MLP 反向传播在 `algorithms/nn.ts`。

> 注意：`family` 只是元信息。**实际用哪个可视化器，由 `AlgorithmLab.tsx` 的 `DEMOS` 表里每条的 `Viz` 字段决定**（所以同属 `scatter-boundary` 的回归和分类能用不同 Viz）。

### 两种数据来源

1. **浏览器端 TS 实时计算**（`src/algorithms/*.ts`）：`run*(opts)` 函数纯手写算法，返回 `Trajectory`。可交互（“重新生成数据”换 seed 重算）。无任何 ML 库。
2. **Python 预计算**（`solvers/*.py` → `src/data/frames/*.json`）：训练真实模型后导帧，前端 `import` JSON 回放。用于 RL / CNN / Transformer（训练太重，浏览器跑不动）。

---

## 4. 怎么加一个新算法

### 4A. 浏览器端算法（监督/无监督/小模型）

1. 在 `src/algorithms/foo.ts` 写 `export function runFoo(opts): Trajectory<XxxState>`，每个迭代 push 一帧。
   - 数据集复用 `algorithms/datasets.ts`；随机数用 `algorithms/rng.ts`（seed 可复现）；分类边界网格用 `algorithms/grid.ts`。
   - `meta` 里**务必填 `tutorial`**（五段式：problem/intuition/watch/concepts/tryThis），这是教学核心。
2. 如果是新家族，在 `types.ts` 加 State 类型 + `Family` 联合类型，并在 `src/visualizers/` 写对应可视化器。否则复用现有的。
3. 在 [`src/pages/AlgorithmLab.tsx`](rl-lab/src/pages/AlgorithmLab.tsx) 的 `DEMOS` 数组加一条，**并把它的 key 挂到 `TREE`（AI 发展脉络树）对应的方法谱系分支下**（否则不会出现在左侧导航）：
   ```ts
   {
     key: "foo", label: "Foo 算法", group: "监督 · 分类",
     build: (seed) => runFoo({ seed }),
     Viz: BoundaryPlot,
     metricKey: "loss", metricLabel: "Loss",
     source: { code: fooSrc, path: "algorithms/foo.ts" },  // 见下
   }
   ```
4. 顶部 `import fooSrc from "@/algorithms/foo.ts?raw"`（`?raw` 把源码当字符串，供「查看源码」面板展示）。

完成。播放器、教程面板、源码查看器、指标曲线全部自动复用。

### 4B. Python 预计算算法（深度学习 / RL）

1. 写 `solvers/your_script.py`，构造 `frames`（list of `{iter, state, metrics}`）和 `meta`（含 `family`、`tutorial`），调用：
   ```python
   from frames_io import write_trajectory      # 见 solvers/frames_io.py
   write_trajectory("your-id", meta, frames)   # → rl-lab/src/data/frames/your-id.json
   ```
   - RL 专用：`frames_io.maybe_record(algorithm, task, meta)` 在 `RECORD_FRAMES=1` 时用 Collector 跑一条 episode 自动导帧（见任一 `solve_*.py` 结尾）。
2. 前端：`import yourData from "@/data/frames/your-id.json"`，在 `DEMOS` 加一条 `build: () => yourData as unknown as Trajectory`，配上对应 `Viz`。
3. 新家族同 4A 第 2 步（加 State 类型 + 可视化器）。

> **教程同步坑**：RL 的 `tutorial` 同时存在于 Python `meta` 和已生成的 JSON 里。重跑 `RECORD_FRAMES=1` 会覆盖 JSON，所以两边内容要保持一致（CNN/Transformer 只有 Python 一处，无此问题）。

---

## 5. 已知的坑（务必知道）

- **端口 5180 写死**（`vite.config.ts` strictPort）。被占用会启动失败，先 `pkill -f vite`。
- **开发时 HMR 偶尔把路由重置回首页 `/`**。用浏览器控制台测 `/lab` 时，如果发现 `location.pathname` 变了，直接 `location.href='/lab'` 重新导航再测。
- **`.gitignore` 只忽略根目录 `/frames/`**，不忽略 `rl-lab/src/data/frames/`（样例帧要入库）。别加宽泛的 `frames/` 规则，会误杀。
- **JSON 导入**需要 `tsconfig.json` 的 `resolveJsonModule: true`（已开），且导入后 `as unknown as Trajectory` 转型。
- **`?raw` 导入**靠 `vite/client` 的类型声明（`src/vite-env.d.ts`），无需额外 d.ts。
- **bundle > 500kB 警告**：prism 高亮库 + 内联源码字符串所致，学习 lab 可接受，别花时间消警告。
- **`index.css` 有个 `@import must precede...` 警告**：字体 @import 写在 @tailwind 之后导致，预先存在、无害。
- **路由结构**（`App.tsx`）：只有 `/` 和 `/lab` = 实验台 `AlgorithmLab`，是唯一入口。所有实验（含 RL）都统一在这里，左侧导航是一棵 **AI 发展脉络树**（`AlgorithmLab.tsx` 的 `TREE` 常量：分支=方法谱系如「价值法/策略法/Actor-Critic」，叶子=`demo.key`，`era` 字段标年代；`renderTree` 递归渲染）。**加新实验记得把 key 挂到 `TREE` 对应分支**，否则不出现在侧栏。
  - （早期那套独立 RL 卡片看板 `Home.tsx`/`ExperimentDetail.tsx`/`data/experiments.ts` 及 `*Viz` 组件已删除——RL 已并入统一实验台。）

---

## 6. 验证流程（改完代码必做）

```bash
cd rl-lab
npx tsc --noEmit        # 类型
npx vite build          # 构建
```
再起 dev server 或用 preview，**真的在浏览器里点一遍**改动的算法（拖到末帧，看动画/边界/热图是否正确）。本项目一直坚持"实跑验证"，不要只靠 tsc 通过就算完。

---

## 7. 当前状态与待办

**已完成**：M1 脚手架 ✅ · M2 监督/无监督（8 个浏览器算法）✅ · M3 RL 接入（CartPole/MountainCar/Pendulum 回放）✅ · M4 深度学习（MLP + CNN + Self-Attention）✅

`/lab` 现有 **26 个 demo**，按 AI 发展时间线覆盖 1958→2023：
- 监督：线性/逻辑回归、感知机、SVM、决策树、AdaBoost、KNN
- 无监督：K-Means、DBSCAN、PCA
- 联想记忆：Hopfield(1982)
- 深度学习：MLP、RNN、CNN、Self-Attention、Mamba、Word2Vec
- 生成模型：GAN、Diffusion
- 经典控制：CartPole 状态反馈
- 强化学习：Q-Learning(GridWorld 表格法)、CartPole·PPO、MountainCar·DQN/PPO/Shaped、Pendulum·SAC

> 这 8 个里程碑实验（SVM/AdaBoost/Hopfield/RNN/Word2Vec/GAN/Diffusion/Q-Learning）全部**浏览器端手写**（含训练），无任何 ML 库。

**可继续的方向（按价值排序）**：
- 🐍 经典 ML 补全：SVM、随机森林、GBDT、GMM(EM)、t-SNE、层次聚类（走 4B 管线）
- RL 的 Python 源码也接入「查看源码」（需配 Vite 允许读 `src` 外文件，或把源码 `?raw` 拷进 src）
- 补 MountainCar-DQN（失败案例）/ Shaped 的回放帧（脚本已接 `RECORD_FRAMES`，跑一下即可）
- 首页 rebrand：从「Tianshou RL Lab」改成覆盖四大类的定位
- 注意力可加「多头 / 多层」展示，更贴近真实 Transformer

---

## 8. 关键文件索引

| 文件 | 作用 |
|------|------|
| `rl-lab/src/player/types.ts` | ★ 帧契约 + 所有 State 类型 + Family 联合类型 |
| `rl-lab/src/pages/AlgorithmLab.tsx` | ★ `/lab` 核心，`DEMOS` 注册表（加算法主要改这里） |
| `rl-lab/src/player/useTrajectory.ts` | 播放逻辑（含 index 越界钳制，切换更短轨迹防崩） |
| `rl-lab/src/algorithms/*.ts` | 浏览器端手写算法 |
| `rl-lab/src/visualizers/*.tsx` | 各家族可视化器 |
| `rl-lab/src/components/TutorialPanel.tsx` | 五段式小白教程面板 |
| `rl-lab/src/components/CodeViewer.tsx` | 折叠源码查看器（prism 高亮） |
| `solvers/frames_io.py` | Python 导帧管线（write_trajectory / maybe_record） |
| `solvers/train_cnn.py` / `train_transformer.py` | CNN / 自注意力训练导帧 |
| `solvers/solve_*.py` | 5 个 Tianshou RL 脚本（含 RECORD_FRAMES 开关） |
| `rl-lab/src/data/frames/*.json` | Python 预计算的帧数据（入库） |
