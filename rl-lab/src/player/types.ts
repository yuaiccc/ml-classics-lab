// 统一的「帧轨迹」契约 —— 所有算法的过程动画都归约到这个结构。
// 设计见 ML_LAB_DESIGN.md §1、§3。

export type Family =
  | "scatter-boundary"
  | "clusters"
  | "curves"
  | "env"
  | "pca"
  | "cnn"
  | "attention";

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
