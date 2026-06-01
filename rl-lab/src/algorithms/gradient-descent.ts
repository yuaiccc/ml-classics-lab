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
      tutorial: {
        problem: "给一堆散点，找一条最能代表它们整体趋势的直线。",
        intuition:
          "一条直线由斜率和截距两个数决定。一开始随便画一条（水平线），算出它离所有点的总误差，再朝“误差变小”的方向一点点调这两个数——这就是梯度下降。",
        watch: [
          "绿色直线从水平开始，逐帧旋转、平移去贴合点云",
          "左上角方程里的斜率/截距数字一直在变",
          "右侧 MSE Loss 曲线一路下降，说明拟合越来越准",
        ],
        concepts: [
          { term: "MSE 均方误差", explain: "每个点到直线竖直距离的平方，再取平均，越小越准" },
          { term: "梯度下降", explain: "沿着让误差下降最快的方向，小步多次地更新参数" },
          { term: "学习率 lr", explain: "每一步迈多大；太大容易震荡，太小收敛很慢" },
        ],
        tryThis: "点“重新生成数据”换一组点，再拖时间轴看直线如何从水平转到贴合。",
      },
      hyperparams: { lr, steps, n, noise },
    },
    frames,
  };
}
