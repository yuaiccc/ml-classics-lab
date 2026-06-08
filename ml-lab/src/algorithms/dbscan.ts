import { Trajectory, Frame, ClustersState } from "@/player/types";

interface P {
  x: number;
  y: number;
}

function gaussianBlobs(): P[] {
  const pts: P[] = [];
  const centers = [
    { cx: 3, cy: 3 },
    { cx: 7, cy: 7 },
    { cx: 3, cy: 7 },
  ];
  for (const c of centers) {
    for (let i = 0; i < 25; i++) {
      pts.push({ x: c.cx + (Math.random() - 0.5) * 2, y: c.cy + (Math.random() - 0.5) * 2 });
    }
  }
  // 噪声点
  for (let i = 0; i < 15; i++) {
    pts.push({ x: Math.random() * 10, y: Math.random() * 10 });
  }
  return pts;
}

function dist(a: P, b: P): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

const UNVISITED = -2;
const NOISE = -1;

export function runDBSCAN(): Trajectory {
  const points = gaussianBlobs();
  const eps = 1.1;
  const minPts = 4;
  const n = points.length;
  const labels = new Array(n).fill(UNVISITED);
  const frames: Frame[] = [];

  const neighbors = (i: number): number[] => {
    const out: number[] = [];
    for (let j = 0; j < n; j++) if (i !== j && dist(points[i], points[j]) <= eps) out.push(j);
    return out;
  };

  const snapshot = (clusters: number): void => {
    const noise = labels.filter((l) => l === NOISE).length;
    frames.push({
      iter: frames.length,
      state: {
        family: "clusters",
        data: {
          points: points.map((p, i) => ({ x: p.x, y: p.y, cluster: labels[i] })),
          centroids: [],
          k: clusters,
        } as ClustersState,
      },
      metrics: { clusters, noise },
    });
  };

  snapshot(0);
  let cluster = 0;
  for (let i = 0; i < n; i++) {
    if (labels[i] !== UNVISITED) continue;
    const nbrs = neighbors(i);
    if (nbrs.length < minPts) {
      labels[i] = NOISE;
      continue;
    }
    labels[i] = cluster;
    const seeds = [...nbrs];
    let snapCounter = 0;
    for (let s = 0; s < seeds.length; s++) {
      const q = seeds[s];
      if (labels[q] === NOISE) labels[q] = cluster;
      if (labels[q] !== UNVISITED) continue;
      labels[q] = cluster;
      const qn = neighbors(q);
      if (qn.length >= minPts) for (const x of qn) if (!seeds.includes(x)) seeds.push(x);
      if (++snapCounter % 3 === 0) snapshot(cluster + 1);
    }
    cluster++;
    snapshot(cluster);
  }
  // 收尾帧
  snapshot(cluster);

  return {
    meta: {
      id: "dbscan",
      title: "DBSCAN 密度聚类",
      algorithm: "DBSCAN",
      category: "unsupervised",
      source: "browser",
      abstract:
        "解决「簇数量未知 + 任意形状 + 有噪声」的聚类问题：不像 K-Means 需要预先指定 K，DBSCAN 从核心点出发沿密度可达关系扩张，自动发现任意形状的簇，并把低密度区域标为噪声。",
      description: "从核心点出发做密度可达扩张，逐帧吞并邻域点形成簇；密度不足的点被标为噪声（灰色）。",
      hyperparams: { eps, minPts, samples: points.length },
      insight:
        "DBSCAN 的核心概念：核心点（邻域内点数 ≥ minPts）、密度可达、噪声。优点是无需指定簇数、能发现任意形状；缺点是对 eps/minPts 敏感，且不适合密度差异很大的数据。",
    },
    frames,
  };
}
