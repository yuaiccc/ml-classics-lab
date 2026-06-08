// 从 ml-lab（Tianshou 演进版）搬来的真实数据集深度学习实验所用的 state 形状。
// 这些实验的帧由 Python 预计算，经 bridgeMlLab 解包后直接喂给下面的可视化器。

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

export interface AttentionState {
  tokens: string[];
  // attention[layer][head] 是一个 L×L 矩阵
  attention: number[][][][];
}
