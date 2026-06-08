import { Trajectory, Frame, ScatterBoundaryState } from "@/player/types";

interface P {
  x: number;
  y: number;
  label: number;
}

interface Leaf {
  x0: number;
  x1: number;
  y0: number;
  y1: number;
  pts: P[];
}

const xRange: [number, number] = [0, 10];
const yRange: [number, number] = [0, 10];
const gridX = 50;
const gridY = 50;

function genXOR(): P[] {
  // 四象限 XOR：对角同类 → 需要多次轴对齐切分
  const pts: P[] = [];
  const quads = [
    { cx: 2.5, cy: 2.5, label: 0 },
    { cx: 7.5, cy: 7.5, label: 0 },
    { cx: 2.5, cy: 7.5, label: 1 },
    { cx: 7.5, cy: 2.5, label: 1 },
  ];
  for (const q of quads) {
    for (let i = 0; i < 18; i++) {
      pts.push({ x: q.cx + (Math.random() - 0.5) * 3, y: q.cy + (Math.random() - 0.5) * 3, label: q.label });
    }
  }
  return pts;
}

function gini(pts: P[]): number {
  if (pts.length === 0) return 0;
  const p1 = pts.filter((p) => p.label === 1).length / pts.length;
  return 1 - p1 * p1 - (1 - p1) * (1 - p1);
}

function majority(pts: P[]): number {
  const ones = pts.filter((p) => p.label === 1).length;
  return ones >= pts.length - ones ? 1 : 0;
}

interface Split {
  axis: "x" | "y";
  thr: number;
  gain: number;
}

function bestSplit(leaf: Leaf): Split | null {
  let best: Split | null = null;
  const base = gini(leaf.pts) * leaf.pts.length;
  for (const axis of ["x", "y"] as const) {
    const vals = [...new Set(leaf.pts.map((p) => p[axis]))].sort((a, b) => a - b);
    for (let i = 0; i < vals.length - 1; i++) {
      const thr = (vals[i] + vals[i + 1]) / 2;
      const left = leaf.pts.filter((p) => p[axis] <= thr);
      const right = leaf.pts.filter((p) => p[axis] > thr);
      if (left.length === 0 || right.length === 0) continue;
      const gain = base - gini(left) * left.length - gini(right) * right.length;
      if (!best || gain > best.gain) best = { axis, thr, gain };
    }
  }
  return best;
}

export function runDecisionTree(): Trajectory {
  const points = genXOR();
  const maxSplits = 8;
  let leaves: Leaf[] = [
    { x0: xRange[0], x1: xRange[1], y0: yRange[0], y1: yRange[1], pts: points },
  ];
  const frames: Frame[] = [];

  const predictAt = (x: number, y: number): number => {
    for (const lf of leaves) {
      if (x >= lf.x0 && x <= lf.x1 && y >= lf.y0 && y <= lf.y1) return majority(lf.pts);
    }
    return 0;
  };

  const accuracy = (): number => {
    let c = 0;
    for (const p of points) if (predictAt(p.x, p.y) === p.label) c++;
    return c / points.length;
  };

  const snapshot = (): void => {
    const boundary: number[][] = [];
    for (let i = 0; i < gridX; i++) {
      boundary[i] = [];
      const gx = xRange[0] + (i / gridX) * (xRange[1] - xRange[0]);
      for (let j = 0; j < gridY; j++) {
        const gy = yRange[0] + (j / gridY) * (yRange[1] - yRange[0]);
        boundary[i][j] = predictAt(gx, gy);
      }
    }
    frames.push({
      iter: frames.length,
      state: {
        family: "scatter-boundary",
        data: { points, boundary, gridX, gridY, xRange, yRange } as ScatterBoundaryState,
      },
      metrics: { splits: leaves.length - 1, accuracy: Number(accuracy().toFixed(4)) },
    });
  };

  snapshot();
  for (let step = 0; step < maxSplits; step++) {
    // 选择增益最大的叶子进行切分
    let target = -1;
    let targetSplit: Split | null = null;
    leaves.forEach((lf, idx) => {
      if (lf.pts.length < 4 || gini(lf.pts) === 0) return;
      const sp = bestSplit(lf);
      if (sp && sp.gain > 0 && (!targetSplit || sp.gain > targetSplit.gain)) {
        target = idx;
        targetSplit = sp;
      }
    });
    if (target < 0 || !targetSplit) break;
    const lf = leaves[target];
    const sp: Split = targetSplit;
    const left: Leaf =
      sp.axis === "x"
        ? { x0: lf.x0, x1: sp.thr, y0: lf.y0, y1: lf.y1, pts: lf.pts.filter((p) => p.x <= sp.thr) }
        : { x0: lf.x0, x1: lf.x1, y0: lf.y0, y1: sp.thr, pts: lf.pts.filter((p) => p.y <= sp.thr) };
    const right: Leaf =
      sp.axis === "x"
        ? { x0: sp.thr, x1: lf.x1, y0: lf.y0, y1: lf.y1, pts: lf.pts.filter((p) => p.x > sp.thr) }
        : { x0: lf.x0, x1: lf.x1, y0: sp.thr, y1: lf.y1, pts: lf.pts.filter((p) => p.y > sp.thr) };
    leaves = [...leaves.slice(0, target), left, right, ...leaves.slice(target + 1)];
    snapshot();
  }

  return {
    meta: {
      id: "decision-tree",
      title: "决策树分类",
      algorithm: "Decision Tree",
      category: "supervised",
      source: "browser",
      abstract:
        "用一系列「是非问题」把空间递归切成矩形区域：每次选一个特征和阈值，使切分后子区域尽可能纯（gini 最小）。本演示用 XOR 型数据，逐帧增加一次切分，看决策边界如何阶梯式逼近。",
      description: "贪心地逐次选择增益最大的叶子做轴对齐切分，决策边界逐帧细化为矩形拼块。",
      hyperparams: { criterion: "gini", max_splits: maxSplits, samples: points.length },
      insight:
        "决策树用轴对齐切分递归划分空间，天然可解释。它能拟合 XOR 这类线性不可分数据（线性模型做不到），但容易过拟合——切分越多训练精度越高，泛化却可能下降，需要剪枝或限制深度。",
    },
    frames,
  };
}
