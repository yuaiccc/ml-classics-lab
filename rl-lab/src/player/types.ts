// 统一的「帧轨迹」契约 —— 所有算法的过程动画都归约到这个结构。
// 设计见 ML_LAB_DESIGN.md §1、§3。

export type Family =
  | "scatter-boundary"
  | "clusters"
  | "curves"
  | "env"
  | "pca"
  | "cnn"
  | "attention"
  | "ssm"
  | "gridworld"
  | "hopfield"
  | "rnn"
  | "embedding"
  | "multiclass"
  | "curvefit"
  | "roc";

/** 小白教程内容：把每个算法做成一节可读的小课。 */
export interface Tutorial {
  /** 一句话：这个算法解决什么问题 */
  problem: string;
  /** 大白话直觉理解 */
  intuition: string;
  /** 看动画时该注意什么（要点列表，引导观察） */
  watch: string[];
  /** 关键概念名词解释 */
  concepts: { term: string; explain: string }[];
  /** 鼓励动手交互的提示 */
  tryThis?: string;
}

export interface TrajectoryMeta {
  id: string;
  title: string;
  family: Family;
  algorithm: string;
  description?: string;
  hyperparams?: Record<string, string | number>;
  /** 教学要点：这个算法在动画里到底在“干什么”（tutorial 缺省时的回退） */
  insight?: string;
  /** 小白教程（优先于 insight 渲染） */
  tutorial?: Tutorial;
  /** env 家族：指明用哪个环境渲染器（CartPole-v1 / MountainCar-v0 / Pendulum-v1） */
  envId?: string;
}

/** 时间轴上的一帧：算法在第 iter 步的可画状态 + 标量指标 */
export interface Frame<S = unknown> {
  iter: number;
  state: S;
  metrics?: Record<string, number>;
}

export interface Trajectory<S = unknown> {
  meta: TrajectoryMeta;
  frames: Frame<S>[];
}

// ---- 家族相关的 state 形状 ----

export interface Point2D {
  x: number;
  y: number;
}

/** scatter-boundary（回归变体）：数据点 + 当前拟合直线 */
export interface RegressionState {
  points: Point2D[];
  fit: { slope: number; intercept: number };
}

export interface ClusterPoint extends Point2D {
  cluster: number;
}

/** clusters：点 + 当前归属 + 当前质心。cluster = -1 表示噪声点（DBSCAN）。 */
export interface ClusterState {
  points: ClusterPoint[];
  centroids: Point2D[];
}

export interface LabeledPoint extends Point2D {
  label: number;
}

/** 决策边界网格：row-major，values[r*cols+c] ∈ [0,1] 表示该格点预测为类别 1 的得分/概率 */
export interface BoundaryGrid {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  cols: number;
  rows: number;
  values: number[];
}

/** classification 家族：带标签的点 + 决策边界网格（+ 可选线性边界叠加） */
export interface BoundaryState {
  points: LabeledPoint[];
  grid: BoundaryGrid;
  line?: { slope: number; intercept: number };
}

/** pca 家族：数据点 + 均值 + 当前估计的主成分方向（单位向量） */
export interface PCAState {
  points: Point2D[];
  mean: Point2D;
  axis: Point2D;
  /** 上一帧的主轴，用于画收敛轨迹（可选） */
  prevAxis?: Point2D;
}

/** env 家族：RL 环境某一步的观测 + 动作（由 Python 录制管线导出） */
export interface EnvState {
  observation: number[];
  action?: number | number[];
}

/** cnn 家族：某一训练 epoch 的卷积核 + 一个样本的激活图（Python 导出） */
export interface CnnState {
  input: number[][]; // H×W 输入图
  filters: number[][][]; // [F][k][k] 第一层卷积核
  activations: number[][][]; // [F][h][w] 该样本的激活图
  label: number;
  pred: number;
}

/** attention 家族：某一训练 epoch 的自注意力权重矩阵（Python 导出） */
export interface AttentionState {
  tokens: number[]; // 输入 token id 序列
  target: number[]; // 目标序列
  pred: number[]; // 模型预测序列
  attention: number[][]; // L×L 注意力权重（行=输出/查询位，列=输入/键位）
}

/** ssm 家族（Mamba 选择性扫描）：浏览器端实时计算，时间轴 = 扫描到第 pos 个位置 */
export interface SSMState {
  tokens: number[]; // 整个序列（0=空白，>0=信号 token）
  pos: number; // 当前扫描位置
  gates: number[]; // 每个位置的选择门 Δ ∈ [0,1]（input-dependent）
  memory: number[][]; // [L][C] 每个时刻的记忆向量（C 个信号通道）
  output: number[]; // 每个时刻输出的 token id（=记忆里最强的信号）
}

/** gridworld 家族（表格 Q-Learning）：价值热图 + 贪心策略箭头 */
export interface GridWorldState {
  rows: number;
  cols: number;
  values: number[]; // 每格 V(s)=max_a Q(s,a)
  policy: number[]; // 每格贪心动作 0=上 1=右 2=下 3=左，终止格=-1
  goal: number; // 目标格索引
  pit: number; // 陷阱格索引
}

/** hopfield 家族：N×N 二值图案的联想记忆回忆过程 */
export interface HopfieldState {
  size: number; // 图案边长 N
  cells: number[]; // 当前状态，长度 N*N，值 ∈ {-1,+1}
  target: number[]; // 想回忆出的目标图案
}

/** rnn 家族：某一训练 epoch 下，一个样本序列的隐藏状态轨迹 */
export interface RnnState {
  inputs: number[]; // 输入序列（0/1 比特）
  hidden: number[][]; // [T][H] 每步隐藏状态
  target: number; // 目标（如奇偶）
  pred: number; // 预测
}

/** embedding 家族（Word2Vec）：词向量 2D 位置（随训练移动） */
export interface EmbeddingState {
  words: string[]; // 词
  positions: { x: number; y: number }[]; // 对应 2D 向量
  groups: number[]; // 词所属语义组（着色用）
}

/** multiclass 家族：真实数据多分类的决策边界（grid.values = 预测的类别索引） */
export interface MultiBoundaryState {
  points: { x: number; y: number; label: number }[];
  grid: BoundaryGrid; // values 存预测类别索引 0/1/2...
  classNames: string[];
  xName: string;
  yName: string;
}

/** curvefit 家族：多项式拟合（过拟合/正则化）—— 训练点 + 拟合曲线 + 真实函数 */
export interface CurveFitState {
  train: { x: number; y: number }[];
  test: { x: number; y: number }[];
  fit: { x: number; y: number }[]; // 采样的拟合曲线
  truth: { x: number; y: number }[]; // 真实函数曲线
  caption: string; // 当前阶数 / λ 等说明
}

/** roc 家族：分类阈值评估 —— 分数分布 + ROC 曲线 + 混淆矩阵 */
export interface RocState {
  threshold: number;
  pos: number[]; // 正类样本分数
  neg: number[]; // 负类样本分数
  roc: { fpr: number; tpr: number }[]; // 完整 ROC 曲线
  current: { fpr: number; tpr: number };
  confusion: { tp: number; fp: number; fn: number; tn: number };
  precision: number;
  recall: number;
  f1: number;
  auc: number;
}
