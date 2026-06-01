// 统一的「帧轨迹」契约 —— 所有算法的过程动画都归约到这个结构。
// 设计见 ML_LAB_DESIGN.md §1、§3。

export type Family = "scatter-boundary" | "clusters" | "curves" | "env" | "pca";

export interface TrajectoryMeta {
  id: string;
  title: string;
  family: Family;
  algorithm: string;
  description?: string;
  hyperparams?: Record<string, string | number>;
  /** 教学要点：这个算法在动画里到底在“干什么” */
  insight?: string;
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
