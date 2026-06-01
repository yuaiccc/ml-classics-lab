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
      tutorial: {
        problem: "同样是把两类点用一条直线分开——这是最古老的线性分类器。",
        intuition:
          "它不看概率，只问“分对了没有”。每遇到一个被分错的点，就把分界线朝那个点的方向“推”一下。只要两类本来就能被一条直线分开，推有限次后必然全部分对。",
        watch: [
          "只有遇到分错的点才更新，边界是一跳一跳地移动（不像逻辑回归那样平滑）",
          "右侧“误分类点数”阶梯式下降",
          "降到 0 之后边界就不再动了——这就是收敛",
        ],
        concepts: [
          { term: "线性可分", explain: "存在一条直线能把两类点完全分开" },
          { term: "在线更新", explain: "一个样本一个样本地修正，而不是一次看全部" },
          { term: "收敛", explain: "误分类降为 0，参数不再变化" },
        ],
        tryThis: "用播放器的“下一步”按钮单步走，观察每一步是哪个点触发了边界移动。",
      },
      hyperparams: { lr, maxUpdates },
    },
    frames,
  };
}
