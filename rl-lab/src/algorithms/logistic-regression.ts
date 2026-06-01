// 逻辑回归（梯度下降）：每步记录决策边界网格 + 交叉熵 / 准确率。
import { Trajectory, BoundaryState, Frame } from "@/player/types";
import { makeBlobs } from "./datasets";
import { computeBounds, makeGrid, sigmoid } from "./grid";

export interface LogRegOptions {
  seed?: number;
  steps?: number;
  lr?: number;
}

export function runLogisticRegression(opts: LogRegOptions = {}): Trajectory<BoundaryState> {
  const { seed = (Date.now() & 0xffff) >>> 0, steps = 50, lr = 0.4 } = opts;
  const points = makeBlobs(seed);
  const bounds = computeBounds(points);

  let w1 = 0;
  let w2 = 0;
  let b = 0;
  const frames: Frame<BoundaryState>[] = [];

  for (let i = 0; i <= steps; i++) {
    const score = (x: number, y: number) => sigmoid(w1 * x + w2 * y + b);
    const grid = makeGrid(bounds, 40, 30, score);

    let loss = 0;
    let correct = 0;
    for (const p of points) {
      const ph = sigmoid(w1 * p.x + w2 * p.y + b);
      loss += -(p.label * Math.log(ph + 1e-9) + (1 - p.label) * Math.log(1 - ph + 1e-9));
      if ((ph >= 0.5 ? 1 : 0) === p.label) correct++;
    }
    loss /= points.length;

    // 决策边界 w1·x + w2·y + b = 0  → y = -(w1/w2)x - b/w2
    const line =
      Math.abs(w2) > 1e-6 ? { slope: -w1 / w2, intercept: -b / w2 } : undefined;

    frames.push({
      iter: i,
      state: { points, grid, line },
      metrics: { loss, accuracy: correct / points.length },
    });

    // 梯度（交叉熵对 w, b）
    let g1 = 0;
    let g2 = 0;
    let gb = 0;
    for (const p of points) {
      const e = sigmoid(w1 * p.x + w2 * p.y + b) - p.label;
      g1 += e * p.x;
      g2 += e * p.y;
      gb += e;
    }
    const n = points.length;
    w1 -= (lr * g1) / n;
    w2 -= (lr * g2) / n;
    b -= (lr * gb) / n;
  }

  return {
    meta: {
      id: "logreg",
      title: "逻辑回归",
      family: "scatter-boundary",
      algorithm: "Logistic Regression",
      description: "两个高斯团簇的二分类，sigmoid + 梯度下降逐步学出线性决策边界。",
      insight:
        "权重从 0 出发，决策边界（白线）逐帧旋转、平移到把两类分开。背景色是预测概率场：越蓝越判为类 1，分界处概率 0.5。loss 下降、准确率上升至饱和。",
      hyperparams: { lr, steps },
    },
    frames,
  };
}
