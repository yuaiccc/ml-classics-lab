// 梯度下降做线性回归：每一步记录当前拟合直线 + MSE。
// 浏览器原生实现，可交互重算。产出统一的 Trajectory（见 player/types.ts）。

import { Trajectory, RegressionState, Frame } from "@/player/types";
import { mulberry32, gaussian } from "./rng";

export interface LinRegOptions {
  n?: number;
  trueSlope?: number;
  trueIntercept?: number;
  noise?: number;
  lr?: number;
  steps?: number;
  seed?: number;
}

export function runLinearRegressionGD(
  opts: LinRegOptions = {}
): Trajectory<RegressionState> {
  const {
    n = 40,
    trueSlope = 1.8,
    trueIntercept = -0.5,
    noise = 1.2,
    lr = 0.04,
    steps = 60,
    seed = (Date.now() & 0xffff) >>> 0,
  } = opts;

  const rng = mulberry32(seed);
  const points = Array.from({ length: n }, () => {
    const x = (rng() * 2 - 1) * 5; // x ∈ [-5, 5]
    const y = trueSlope * x + trueIntercept + gaussian(rng, 0, noise);
    return { x, y };
  });

  const mse = (w: number, b: number) =>
    points.reduce((s, p) => {
      const e = w * p.x + b - p.y;
      return s + e * e;
    }, 0) / points.length;

  let w = 0;
  let b = 0;
  const frames: Frame<RegressionState>[] = [];

  for (let i = 0; i <= steps; i++) {
    frames.push({
      iter: i,
      state: { points, fit: { slope: w, intercept: b } },
      metrics: { loss: mse(w, b) },
    });

    // MSE 对 (w, b) 的梯度
    let gw = 0;
    let gb = 0;
    for (const p of points) {
      const e = w * p.x + b - p.y;
      gw += 2 * e * p.x;
      gb += 2 * e;
    }
    gw /= points.length;
    gb /= points.length;
    w -= lr * gw;
    b -= lr * gb;
  }

  return {
    meta: {
      id: "linreg-gd",
      title: "线性回归 · 梯度下降",
      family: "scatter-boundary",
      algorithm: "Gradient Descent",
      description: `拟合带噪声的 y = ${trueSlope}x + ${trueIntercept}，梯度下降逐步逼近真实直线。`,
      insight:
        "权重从 (0, 0) 出发，每步沿损失下降最快的方向更新。直线先快速旋转贴合斜率，再微调截距，loss 单调下降直至收敛。",
      hyperparams: { lr, steps, n, noise },
    },
    frames,
  };
}
