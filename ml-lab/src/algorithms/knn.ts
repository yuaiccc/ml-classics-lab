import { Trajectory, Frame, ScatterBoundaryState } from "@/player/types";

interface P {
  x: number;
  y: number;
  label: number;
}

function genData(): P[] {
  const pts: P[] = [];
  const blobs = [
    { cx: 3, cy: 3, label: 0 },
    { cx: 7, cy: 6, label: 1 },
    { cx: 3.5, cy: 7, label: 1 },
  ];
  for (const b of blobs) {
    for (let i = 0; i < 22; i++) {
      pts.push({ x: b.cx + (Math.random() - 0.5) * 2.4, y: b.cy + (Math.random() - 0.5) * 2.4, label: b.label });
    }
  }
  return pts;
}

const xRange: [number, number] = [0, 10];
const yRange: [number, number] = [0, 10];
const gridX = 50;
const gridY = 50;

export function runKNN(): Trajectory {
  const points = genData();
  const kValues = [1, 3, 5, 9, 15, 25];
  const frames: Frame[] = [];

  const classify = (x: number, y: number, k: number): number => {
    const sorted = points
      .map((p) => ({ d: (p.x - x) ** 2 + (p.y - y) ** 2, label: p.label }))
      .sort((a, b) => a.d - b.d)
      .slice(0, k);
    let c0 = 0;
    for (const s of sorted) if (s.label === 0) c0++;
    return c0 > k / 2 ? 0 : 1;
  };

  const accuracyFor = (k: number): number => {
    let correct = 0;
    for (const p of points) {
      // leave-one-out 近似：直接分类（含自身），k 较大时影响小
      if (classify(p.x, p.y, k) === p.label) correct++;
    }
    return correct / points.length;
  };

  for (const k of kValues) {
    const boundary: number[][] = [];
    for (let i = 0; i < gridX; i++) {
      boundary[i] = [];
      const gx = xRange[0] + (i / gridX) * (xRange[1] - xRange[0]);
      for (let j = 0; j < gridY; j++) {
        const gy = yRange[0] + (j / gridY) * (yRange[1] - yRange[0]);
        boundary[i][j] = classify(gx, gy, k);
      }
    }
    frames.push({
      iter: k,
      state: {
        family: "scatter-boundary",
        data: { points, boundary, gridX, gridY, xRange, yRange } as ScatterBoundaryState,
      },
      metrics: { k, accuracy: Number(accuracyFor(k).toFixed(4)) },
    });
  }

  return {
    meta: {
      id: "knn",
      title: "KNN 最近邻分类",
      algorithm: "K-Nearest Neighbors",
      category: "supervised",
      source: "browser",
      abstract:
        "最简单的分类思想：一个点的类别由它最近的 K 个邻居投票决定，没有显式训练。本演示让 K 从 1 增大到 25，观察决策边界如何从锯齿状（过拟合）逐渐变平滑（欠拟合）。",
      description: "非迭代算法。逐帧增大 K，展示决策边界随 K 从碎片化到平滑的偏差-方差权衡。",
      hyperparams: { k_values: kValues.join(","), samples: points.length, metric: "euclidean" },
      insight:
        "KNN 是惰性学习：不训练，预测时才计算。K 小 → 边界贴合训练点但对噪声敏感（高方差）；K 大 → 边界平滑但可能忽略局部结构（高偏差）。这是偏差-方差权衡最直观的例子。",
    },
    frames,
  };
}
