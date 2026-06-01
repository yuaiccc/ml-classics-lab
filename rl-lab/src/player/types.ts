// 统一的「帧轨迹」契约 —— 所有算法的过程动画都归约到这个结构。
// 设计见 ML_LAB_DESIGN.md §1、§3。

export type Family = "scatter-boundary" | "clusters" | "curves" | "env";

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

/** clusters：点 + 当前归属 + 当前质心 */
export interface ClusterState {
  points: ClusterPoint[];
  centroids: Point2D[];
}
