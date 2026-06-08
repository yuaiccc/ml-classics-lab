export type VisualizerFamily = "scatter-boundary" | "clusters" | "curves" | "env" | "attention" | "image-grid";

export interface ScatterBoundaryState {
  points: { x: number; y: number; label: number }[];
  boundary: number[][];
  fit?: { slope: number; intercept: number } | number[];
  gridX: number;
  gridY: number;
  xRange: [number, number];
  yRange: [number, number];
}

export interface ClustersState {
  points: { x: number; y: number; cluster: number }[];
  centroids: { x: number; y: number }[];
  k: number;
}

export interface CurvesState {
  embedding?: { x: number; y: number; label: number }[];
}

export interface EnvState {
  observation: number[];
  action?: number;
  reward?: number;
  envType: string;
}

export interface AttentionState {
  tokens: string[];
  // attention[layer][head] 是一个 L×L 矩阵
  attention: number[][][][];
}

export interface ImageGridGroup {
  title: string;
  w: number;
  h: number;
  colormap: "diverging" | "sequential";
  // 每张 image 是长度 w*h、值 [0,1] 的扁平数组
  images: number[][];
}

export interface ImageGridState {
  groups: ImageGridGroup[];
  // 可选：探针原图（如 MNIST 探针数字）
  probe?: { w: number; h: number; pixels: number[]; label?: number };
}

export type FrameState =
  | { family: "scatter-boundary"; data: ScatterBoundaryState }
  | { family: "clusters"; data: ClustersState }
  | { family: "curves"; data: CurvesState }
  | { family: "env"; data: EnvState }
  | { family: "attention"; data: AttentionState }
  | { family: "image-grid"; data: ImageGridState };

export interface Frame {
  iter: number;
  state: FrameState;
  metrics: Record<string, number>;
}

export interface TrajectoryMeta {
  id: string;
  title: string;
  algorithm: string;
  category: "supervised" | "unsupervised" | "deep" | "rl";
  source: "browser" | "python";
  abstract: string;
  description: string;
  hyperparams: Record<string, string | number>;
  insight: string;
}

export interface Trajectory {
  meta: TrajectoryMeta;
  frames: Frame[];
}
