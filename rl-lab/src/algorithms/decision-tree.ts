// 决策树（CART，Gini）：按深度逐层加深，展示轴对齐边界如何越切越细。
import { Trajectory, BoundaryState, Frame, LabeledPoint } from "@/player/types";
import { makeCircles } from "./datasets";
import { computeBounds, makeGrid } from "./grid";

interface TreeNode {
  pred: number; // 该节点多数类（叶子用）
  feat?: 0 | 1; // 0=x, 1=y
  thresh?: number;
  left?: TreeNode;
  right?: TreeNode;
}

function majority(pts: LabeledPoint[]): number {
  let c1 = 0;
  for (const p of pts) c1 += p.label;
  return c1 * 2 >= pts.length ? 1 : 0;
}

function gini(pts: LabeledPoint[]): number {
  if (pts.length === 0) return 0;
  let c1 = 0;
  for (const p of pts) c1 += p.label;
  const p1 = c1 / pts.length;
  return 1 - p1 * p1 - (1 - p1) * (1 - p1);
}

function build(pts: LabeledPoint[], depth: number, maxDepth: number): TreeNode {
  const node: TreeNode = { pred: majority(pts) };
  if (depth >= maxDepth || gini(pts) < 1e-9 || pts.length < 4) return node;

  let best = { gain: 0, feat: 0 as 0 | 1, thresh: 0, left: [] as LabeledPoint[], right: [] as LabeledPoint[] };
  const parentGini = gini(pts);
  for (const feat of [0, 1] as const) {
    const vals = [...new Set(pts.map((p) => (feat === 0 ? p.x : p.y)))].sort((a, b) => a - b);
    for (let i = 0; i < vals.length - 1; i++) {
      const thresh = (vals[i] + vals[i + 1]) / 2;
      const left = pts.filter((p) => (feat === 0 ? p.x : p.y) <= thresh);
      const right = pts.filter((p) => (feat === 0 ? p.x : p.y) > thresh);
      if (!left.length || !right.length) continue;
      const w = (left.length * gini(left) + right.length * gini(right)) / pts.length;
      const gain = parentGini - w;
      if (gain > best.gain) best = { gain, feat, thresh, left, right };
    }
  }
  if (best.gain <= 1e-9) return node;
  node.feat = best.feat;
  node.thresh = best.thresh;
  node.left = build(best.left, depth + 1, maxDepth);
  node.right = build(best.right, depth + 1, maxDepth);
  return node;
}

function predict(node: TreeNode, x: number, y: number): number {
  if (node.feat === undefined) return node.pred;
  const v = node.feat === 0 ? x : y;
  return predict(v <= node.thresh! ? node.left! : node.right!, x, y);
}

export interface DTreeOptions {
  seed?: number;
  maxDepth?: number;
}

export function runDecisionTree(opts: DTreeOptions = {}): Trajectory<BoundaryState> {
  const { seed = (Date.now() & 0xffff) >>> 0, maxDepth = 6 } = opts;
  const points = makeCircles(seed);
  const bounds = computeBounds(points);
  const frames: Frame<BoundaryState>[] = [];

  for (let d = 0; d <= maxDepth; d++) {
    const tree = build(points, 0, d);
    const grid = makeGrid(bounds, 44, 34, (x, y) => predict(tree, x, y));
    let correct = 0;
    for (const p of points) if (predict(tree, p.x, p.y) === p.label) correct++;
    frames.push({
      iter: d,
      state: { points, grid },
      metrics: { accuracy: correct / points.length },
    });
  }

  return {
    meta: {
      id: "dtree",
      title: "决策树 (CART)",
      family: "scatter-boundary",
      algorithm: "Decision Tree",
      description: "同心圆数据（线性不可分）。每加深一层，用 Gini 增益挑最优轴对齐切分。",
      tutorial: {
        problem: "用一连串“如果…就…”的规则给点分类，连同心圆这种弯曲分布也能处理。",
        intuition:
          "每一步挑一个最能把两类分开的横切或竖切（按 Gini 不纯度评分），把平面切成一块块矩形。切得越细，就越能逼近弯曲的真实边界。",
        watch: [
          "决策边界永远是横平竖直的方块（轴对齐切分）",
          "深度每加 1，就多切几刀，中间橙色区域被逐步框出来",
          "训练准确率随深度上升——但太深会过拟合，把噪声也学进去",
        ],
        concepts: [
          { term: "Gini 不纯度", explain: "一个区域里两类混得有多乱，越纯越接近 0" },
          { term: "轴对齐切分", explain: "每一刀只沿 x 或 y 方向切" },
          { term: "过拟合", explain: "切太细把噪声也当规律，换数据就失灵" },
        ],
        tryThis: "拖时间轴看深度从 0 到 6，边界如何从一整块逐步细化成环形。",
      },
      hyperparams: { maxDepth, points: points.length },
    },
    frames,
  };
}
