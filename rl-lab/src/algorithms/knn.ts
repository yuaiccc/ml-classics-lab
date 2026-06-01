// KNN：非迭代算法，这里用“k 从小到大”作为过程轴，展示 k 对边界平滑度的影响。
import { Trajectory, BoundaryState, Frame } from "@/player/types";
import { makeCircles } from "./datasets";
import { computeBounds, makeGrid } from "./grid";

export interface KNNOptions {
  seed?: number;
  kValues?: number[];
}

export function runKNN(opts: KNNOptions = {}): Trajectory<BoundaryState> {
  const { seed = (Date.now() & 0xffff) >>> 0, kValues = [1, 3, 5, 9, 15, 25, 41] } = opts;
  const points = makeCircles(seed);
  const bounds = computeBounds(points);
  const frames: Frame<BoundaryState>[] = [];

  // 预排序辅助：给定 (x,y) 返回最近 k 个点里类 1 的比例
  const scoreAt = (x: number, y: number, k: number) => {
    const d = points
      .map((p) => ({ label: p.label, d2: (p.x - x) ** 2 + (p.y - y) ** 2 }))
      .sort((a, b) => a.d2 - b.d2);
    let c1 = 0;
    for (let i = 0; i < k && i < d.length; i++) c1 += d[i].label;
    return c1 / k;
  };

  for (const k of kValues) {
    const grid = makeGrid(bounds, 40, 30, (x, y) => scoreAt(x, y, k));
    // 留一法准确率：预测某点时排除它自己
    let correct = 0;
    for (const q of points) {
      const d = points
        .filter((p) => p !== q)
        .map((p) => ({ label: p.label, d2: (p.x - q.x) ** 2 + (p.y - q.y) ** 2 }))
        .sort((a, b) => a.d2 - b.d2);
      let c1 = 0;
      for (let i = 0; i < k && i < d.length; i++) c1 += d[i].label;
      if ((c1 / k >= 0.5 ? 1 : 0) === q.label) correct++;
    }
    frames.push({
      iter: k,
      state: { points, grid },
      metrics: { accuracy: correct / points.length },
    });
  }

  return {
    meta: {
      id: "knn",
      title: "KNN（k 近邻）",
      family: "scatter-boundary",
      algorithm: "KNN",
      description: "同心圆数据。横轴是 k：看近邻数如何影响决策边界的平滑程度。",
      insight:
        "KNN 不训练，直接看“最近的 k 个邻居谁多”。k=1 边界锯齿、易过拟合（贴着每个点）；k 增大边界变平滑但可能欠拟合。这是偏差-方差权衡最直观的演示。",
      hyperparams: { kRange: `${kValues[0]}–${kValues[kValues.length - 1]}`, points: points.length },
    },
    frames,
  };
}
