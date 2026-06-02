// AdaBoost：把一堆“弱分类器”（决策树桩）加权组合成强分类器。
// 浏览器手写。每加一个树桩，边界精化一点。展示集成学习的威力。
import { Trajectory, BoundaryState, Frame, LabeledPoint } from "@/player/types";
import { makeCircles } from "./datasets";
import { computeBounds, makeGrid } from "./grid";

interface Stump {
  feat: 0 | 1;
  thresh: number;
  polarity: 1 | -1;
  alpha: number;
}

export interface AdaBoostOptions {
  seed?: number;
  rounds?: number;
}

export function runAdaBoost(opts: AdaBoostOptions = {}): Trajectory<BoundaryState> {
  const { seed = (Date.now() & 0xffff) >>> 0, rounds = 14 } = opts;
  const points = makeCircles(seed, 50);
  const bounds = computeBounds(points);
  const y = points.map((p) => (p.label === 1 ? 1 : -1));
  const n = points.length;
  let w = new Array(n).fill(1 / n);
  const stumps: Stump[] = [];

  const stumpPred = (s: Stump, p: LabeledPoint) =>
    ((s.feat === 0 ? p.x : p.y) <= s.thresh ? 1 : -1) * s.polarity;
  const ensemble = (x: number, yy: number) => {
    let sum = 0;
    for (const s of stumps) sum += s.alpha * (((s.feat === 0 ? x : yy) <= s.thresh ? 1 : -1) * s.polarity);
    return sum;
  };

  const frames: Frame<BoundaryState>[] = [];
  const snapshot = (iter: number) => {
    const grid = makeGrid(bounds, 44, 34, (x, yy) => (ensemble(x, yy) >= 0 ? 1 : 0));
    let correct = 0;
    for (let i = 0; i < n; i++) if ((ensemble(points[i].x, points[i].y) >= 0 ? 1 : -1) === y[i]) correct++;
    frames.push({ iter, state: { points, grid }, metrics: { accuracy: correct / n, learners: stumps.length } });
  };
  snapshot(0);

  for (let r = 0; r < rounds; r++) {
    // 找加权误差最小的决策树桩
    let best = { err: Infinity, feat: 0 as 0 | 1, thresh: 0, polarity: 1 as 1 | -1 };
    for (const feat of [0, 1] as const) {
      const vals = [...new Set(points.map((p) => (feat === 0 ? p.x : p.y)))].sort((a, b) => a - b);
      for (let i = 0; i < vals.length - 1; i++) {
        const thresh = (vals[i] + vals[i + 1]) / 2;
        for (const polarity of [1, -1] as const) {
          let err = 0;
          for (let k = 0; k < n; k++) {
            const pred = ((feat === 0 ? points[k].x : points[k].y) <= thresh ? 1 : -1) * polarity;
            if (pred !== y[k]) err += w[k];
          }
          if (err < best.err) best = { err, feat, thresh, polarity };
        }
      }
    }
    const eps = Math.max(1e-10, Math.min(1 - 1e-10, best.err));
    const alpha = 0.5 * Math.log((1 - eps) / eps);
    const stump: Stump = { feat: best.feat, thresh: best.thresh, polarity: best.polarity, alpha };
    stumps.push(stump);
    // 更新样本权重：分错的加重
    let Z = 0;
    for (let k = 0; k < n; k++) {
      w[k] *= Math.exp(-alpha * y[k] * stumpPred(stump, points[k]));
      Z += w[k];
    }
    w = w.map((v) => v / Z);
    snapshot(r + 1);
  }

  return {
    meta: {
      id: "adaboost",
      title: "AdaBoost 集成",
      family: "scatter-boundary",
      algorithm: "AdaBoost (decision stumps)",
      description: "同心圆数据。把许多“只切一刀”的弱分类器加权组合，拼出复杂的非线性边界。",
      tutorial: {
        problem: "单个简单模型（只切一刀的树桩）很弱，怎么变强？AdaBoost：把一堆弱的加权组合起来。",
        intuition:
          "每一轮训练一个只切一刀的树桩，专门盯着上一轮“分错的点”（给它们加大权重）。这样后面的树桩不断补前面的漏，加权投票后整体就很强——三个臭皮匠顶个诸葛亮。",
        watch: [
          "每加一个弱分类器（树桩），边界就精化一点",
          "边界由横/竖切片叠加而成，逐渐包出环形",
          "训练准确率随弱分类器数量上升",
        ],
        concepts: [
          { term: "弱分类器", explain: "只比瞎猜好一点的简单模型，这里是单刀决策树桩" },
          { term: "样本重加权", explain: "每轮给分错的样本加大权重，逼后续模型重点关注" },
          { term: "加权投票", explain: "按各弱分类器的准确率 α 加权汇总成强分类器" },
        ],
        tryThis: "拖时间轴看弱分类器从 0 加到十几个，边界怎么一步步逼近环形。",
      },
      hyperparams: { rounds, points: n },
    },
    frames,
  };
}
