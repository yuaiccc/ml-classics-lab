// 感知机：遇到误分类点就更新权重，每次更新记一帧，直观展示边界“被推着走”。
import { Trajectory, BoundaryState, Frame } from "@/player/types";
import { makeBlobs } from "./datasets";
import { computeBounds, makeGrid } from "./grid";

export interface PerceptronOptions {
  seed?: number;
  lr?: number;
  maxUpdates?: number;
}

export function runPerceptron(opts: PerceptronOptions = {}): Trajectory<BoundaryState> {
  const { seed = (Date.now() & 0xffff) >>> 0, lr = 0.5, maxUpdates = 60 } = opts;
  const points = makeBlobs(seed);
  const bounds = computeBounds(points);

  let w1 = 0.2;
  let w2 = -0.3;
  let b = 0;
  const frames: Frame<BoundaryState>[] = [];

  const snapshot = (iter: number) => {
    const score = (x: number, y: number) => (w1 * x + w2 * y + b >= 0 ? 1 : 0);
    const grid = makeGrid(bounds, 40, 30, score);
    let errors = 0;
    for (const p of points) {
      const pred = w1 * p.x + w2 * p.y + b >= 0 ? 1 : 0;
      if (pred !== p.label) errors++;
    }
    const line =
      Math.abs(w2) > 1e-6 ? { slope: -w1 / w2, intercept: -b / w2 } : undefined;
    frames.push({
      iter,
      state: { points, grid, line },
      metrics: { errors },
    });
    return errors;
  };

  snapshot(0);
  let updates = 0;
  let pass = 0;
  while (updates < maxUpdates) {
    let mistakes = 0;
    for (const p of points) {
      const pred = w1 * p.x + w2 * p.y + b >= 0 ? 1 : 0;
      if (pred !== p.label) {
        const sign = p.label === 1 ? 1 : -1; // 标签 {0,1} → 更新方向 {-1,+1}
        w1 += lr * sign * p.x;
        w2 += lr * sign * p.y;
        b += lr * sign;
        mistakes++;
        updates++;
        if (snapshot(updates) === 0 || updates >= maxUpdates) break;
      }
    }
    pass++;
    if (mistakes === 0 || pass > 50) break; // 收敛或兜底
  }

  return {
    meta: {
      id: "perceptron",
      title: "感知机",
      family: "scatter-boundary",
      algorithm: "Perceptron",
      description: "最古老的线性分类器：只在分错的点上更新，把边界往正确方向推。",
      insight:
        "与逻辑回归不同，感知机不看概率，只问“分对没有”。每遇到一个误分类点，就沿该点方向推一下权重（边界）。线性可分时有限步内必然收敛到 0 错误。",
      hyperparams: { lr, maxUpdates },
    },
    frames,
  };
}
