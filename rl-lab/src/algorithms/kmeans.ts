// K-Means（Lloyd 算法）：每一次迭代记录「点的归属 + 质心位置」。
// 浏览器原生实现，可交互重算。产出统一的 Trajectory（见 player/types.ts）。

import { Trajectory, ClusterState, Point2D, Frame } from "@/player/types";
import { mulberry32, gaussian } from "./rng";

export interface KMeansOptions {
  k?: number;
  pointsPerCluster?: number;
  spread?: number;
  maxIter?: number;
  seed?: number;
}

export function runKMeans(opts: KMeansOptions = {}): Trajectory<ClusterState> {
  const {
    k = 3,
    pointsPerCluster = 40,
    spread = 0.8,
    maxIter = 20,
    seed = (Date.now() & 0xffff) >>> 0,
  } = opts;

  const rng = mulberry32(seed);

  // 生成 k 个高斯团簇作为数据
  const trueCenters: Point2D[] = Array.from({ length: k }, () => ({
    x: (rng() * 2 - 1) * 6,
    y: (rng() * 2 - 1) * 6,
  }));
  const raw: Point2D[] = [];
  for (const c of trueCenters) {
    for (let i = 0; i < pointsPerCluster; i++) {
      raw.push({ x: c.x + gaussian(rng, 0, spread), y: c.y + gaussian(rng, 0, spread) });
    }
  }

  // 初始质心：随机挑 k 个数据点
  let centroids: Point2D[] = Array.from({ length: k }, () => {
    const p = raw[Math.floor(rng() * raw.length)];
    return { x: p.x, y: p.y };
  });

  const assign = () =>
    raw.map((p) => {
      let best = 0;
      let bd = Infinity;
      centroids.forEach((c, ci) => {
        const d = (c.x - p.x) ** 2 + (c.y - p.y) ** 2;
        if (d < bd) {
          bd = d;
          best = ci;
        }
      });
      return best;
    });

  const frames: Frame<ClusterState>[] = [];

  for (let it = 0; it <= maxIter; it++) {
    const labels = assign();
    const clustered = raw.map((p, i) => ({ x: p.x, y: p.y, cluster: labels[i] }));

    let inertia = 0;
    clustered.forEach((p) => {
      const c = centroids[p.cluster];
      inertia += (c.x - p.x) ** 2 + (c.y - p.y) ** 2;
    });

    frames.push({
      iter: it,
      state: { points: clustered, centroids: centroids.map((c) => ({ ...c })) },
      metrics: { inertia },
    });

    // 更新质心为各簇均值
    const acc = centroids.map(() => ({ x: 0, y: 0, n: 0 }));
    clustered.forEach((p) => {
      const a = acc[p.cluster];
      a.x += p.x;
      a.y += p.y;
      a.n += 1;
    });
    const next = acc.map((a, ci) => (a.n ? { x: a.x / a.n, y: a.y / a.n } : centroids[ci]));

    let moved = 0;
    next.forEach((c, ci) => {
      moved += Math.abs(c.x - centroids[ci].x) + Math.abs(c.y - centroids[ci].y);
    });
    centroids = next;
    if (it > 0 && moved < 1e-6) break; // 收敛
  }

  return {
    meta: {
      id: "kmeans",
      title: "K-Means 聚类",
      family: "clusters",
      algorithm: "K-Means",
      description: `${k} 个簇，质心随迭代移动直至稳定。`,
      insight:
        "每轮两步：①把每个点分给最近的质心（着色变化）；②把质心移到各簇均值（叉号移动）。inertia（簇内平方和）单调下降，质心不再移动即收敛。",
      hyperparams: { k, points: raw.length, maxIter },
    },
    frames,
  };
}
