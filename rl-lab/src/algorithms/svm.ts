// 线性软间隔 SVM（Pegasos 次梯度法）—— 浏览器手写。
// 和逻辑回归的区别：SVM 找“最大间隔”分界线（离两类都尽量远），更鲁棒。
import { Trajectory, BoundaryState, Frame } from "@/player/types";
import { makeBlobs } from "./datasets";
import { computeBounds, makeGrid, sigmoid } from "./grid";

export interface SVMOptions {
  seed?: number;
  steps?: number;
  lambda?: number;
}

export function runSVM(opts: SVMOptions = {}): Trajectory<BoundaryState> {
  const { seed = (Date.now() & 0xffff) >>> 0, steps = 50, lambda = 0.05 } = opts;
  const points = makeBlobs(seed, 40, 3.6);
  const bounds = computeBounds(points);
  const y = (lab: number) => (lab === 1 ? 1 : -1); // {0,1} -> {-1,+1}

  let w1 = 0;
  let w2 = 0;
  let b = 0;
  const frames: Frame<BoundaryState>[] = [];

  for (let t = 1; t <= steps + 1; t++) {
    const score = (x: number, yy: number) => sigmoid(2 * (w1 * x + w2 * yy + b));
    const grid = makeGrid(bounds, 40, 30, score);
    let correct = 0;
    let hinge = 0;
    for (const p of points) {
      const m = y(p.label) * (w1 * p.x + w2 * p.y + b);
      if (m > 0) correct++;
      hinge += Math.max(0, 1 - m);
    }
    const normW = Math.hypot(w1, w2) || 1e-6;
    const line = Math.abs(w2) > 1e-6 ? { slope: -w1 / w2, intercept: -b / w2 } : undefined;
    frames.push({
      iter: t - 1,
      state: { points, grid, line },
      metrics: { margin: 2 / normW, accuracy: correct / points.length, hinge: hinge / points.length },
    });
    if (t === steps + 1) break;

    // Pegasos 次梯度：min λ/2||w||² + 平均 hinge
    const lr = 1 / (lambda * t);
    let g1 = lambda * w1;
    let g2 = lambda * w2;
    let gb = 0;
    for (const p of points) {
      const m = y(p.label) * (w1 * p.x + w2 * p.y + b);
      if (m < 1) {
        g1 -= (y(p.label) * p.x) / points.length;
        g2 -= (y(p.label) * p.y) / points.length;
        gb -= y(p.label) / points.length;
      }
    }
    w1 -= lr * g1;
    w2 -= lr * g2;
    b -= lr * gb;
  }

  return {
    meta: {
      id: "svm",
      title: "支持向量机 SVM",
      family: "scatter-boundary",
      algorithm: "Linear SVM (Pegasos)",
      description: "两类点，找“最大间隔”的分界线——离两边都尽量远，而不只是分对。",
      tutorial: {
        problem: "能分开两类的直线有无数条，哪条最好？SVM 说：离两类都最远的那条（最大间隔）。",
        intuition:
          "逻辑回归只要分对就行；SVM 还要求分界线两侧留出最宽的“安全带”（间隔）。只有最靠近边界的少数点（支持向量）决定这条线，所以更抗噪、更稳。",
        watch: [
          "分界线（白虚线）逐步调整到让两类间隔最大",
          "右侧“间隔宽度”曲线上升——安全带越拉越宽",
          "和逻辑回归对比：那个只管分对，这个还管离得远不远",
        ],
        concepts: [
          { term: "最大间隔", explain: "分界线到最近样本的距离，越大泛化越好" },
          { term: "支持向量", explain: "贴着间隔边界的少数关键点，只有它们决定分界线" },
          { term: "软间隔 / 核技巧", explain: "软间隔容忍少量越界；核技巧可处理非线性（本例为线性）" },
        ],
        tryThis: "拖时间轴看间隔从窄到宽；点“重新生成数据”换团簇位置。",
      },
      hyperparams: { lambda, steps },
    },
    frames,
  };
}
