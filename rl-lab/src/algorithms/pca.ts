// PCA（幂迭代求第一主成分）：把“求最大方差方向”做成可见的迭代收敛过程。
import { Trajectory, PCAState, Frame } from "@/player/types";
import { makeAnisotropic } from "./datasets";
import { mulberry32 } from "./rng";

export interface PCAOptions {
  seed?: number;
  steps?: number;
}

export function runPCA(opts: PCAOptions = {}): Trajectory<PCAState> {
  const { seed = (Date.now() & 0xffff) >>> 0, steps = 14 } = opts;
  const points = makeAnisotropic(seed);
  const n = points.length;

  const mean = {
    x: points.reduce((s, p) => s + p.x, 0) / n,
    y: points.reduce((s, p) => s + p.y, 0) / n,
  };

  // 2×2 协方差矩阵
  let cxx = 0;
  let cxy = 0;
  let cyy = 0;
  for (const p of points) {
    const dx = p.x - mean.x;
    const dy = p.y - mean.y;
    cxx += dx * dx;
    cxy += dx * dy;
    cyy += dy * dy;
  }
  cxx /= n;
  cxy /= n;
  cyy /= n;

  const mul = (v: { x: number; y: number }) => ({
    x: cxx * v.x + cxy * v.y,
    y: cxy * v.x + cyy * v.y,
  });
  const norm = (v: { x: number; y: number }) => {
    const m = Math.hypot(v.x, v.y) || 1;
    return { x: v.x / m, y: v.y / m };
  };

  const rng = mulberry32(seed ^ 0x9e3779b9);
  let v = norm({ x: rng() * 2 - 1, y: rng() * 2 - 1 }); // 随机初始方向
  const frames: Frame<PCAState>[] = [];

  for (let i = 0; i <= steps; i++) {
    const variance = (() => {
      const cv = mul(v);
      return v.x * cv.x + v.y * cv.y; // Rayleigh 商：沿 v 的方差
    })();
    const prev = i > 0 ? frames[i - 1].state.axis : undefined;
    frames.push({
      iter: i,
      state: { points, mean, axis: { ...v }, prevAxis: prev },
      metrics: { variance },
    });
    v = norm(mul(v)); // 幂迭代：v ← normalize(C v)
  }

  return {
    meta: {
      id: "pca",
      title: "PCA · 幂迭代",
      family: "pca",
      algorithm: "PCA (Power Iteration)",
      description: "各向异性点云。用幂迭代让一个随机方向逐步转到方差最大的主成分方向。",
      insight:
        "PCA 要找“数据散得最开”的方向。幂迭代每步把当前向量乘以协方差矩阵再归一化，方向会不断向最大特征值对应的特征向量（第一主成分）旋转，Rayleigh 商（沿该方向方差）单调上升至收敛。",
      hyperparams: { steps, points: n },
    },
    frames,
  };
}
