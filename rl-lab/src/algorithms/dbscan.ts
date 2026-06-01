// DBSCAN：基于密度的聚类，能发现任意形状的簇并识别噪声。
// 过程轴：每扩展完一个簇记一帧，看簇一个个“长”出来，最后剩下的是噪声。
import { Trajectory, ClusterState, Point2D, Frame } from "@/player/types";
import { mulberry32, gaussian } from "./rng";

export interface DBSCANOptions {
  seed?: number;
  eps?: number;
  minPts?: number;
}

function makeData(seed: number): Point2D[] {
  const rng = mulberry32(seed);
  const centers = [
    { x: -4, y: -3 },
    { x: 4, y: -2 },
    { x: 0, y: 4 },
  ];
  const pts: Point2D[] = [];
  for (const c of centers)
    for (let i = 0; i < 35; i++)
      pts.push({ x: c.x + gaussian(rng, 0, 0.7), y: c.y + gaussian(rng, 0, 0.7) });
  // 均匀噪声
  for (let i = 0; i < 20; i++) pts.push({ x: (rng() * 2 - 1) * 8, y: (rng() * 2 - 1) * 8 });
  return pts;
}

export function runDBSCAN(opts: DBSCANOptions = {}): Trajectory<ClusterState> {
  const { seed = (Date.now() & 0xffff) >>> 0, eps = 1.1, minPts = 4 } = opts;
  const raw = makeData(seed);
  const n = raw.length;
  const labels = new Array(n).fill(-2); // -2=未访问, -1=噪声, >=0 簇号
  const frames: Frame<ClusterState>[] = [];

  const neighbors = (i: number) => {
    const out: number[] = [];
    for (let j = 0; j < n; j++) {
      const dx = raw[i].x - raw[j].x;
      const dy = raw[i].y - raw[j].y;
      if (dx * dx + dy * dy <= eps * eps) out.push(j);
    }
    return out;
  };

  const snapshot = (iter: number) => {
    frames.push({
      iter,
      state: {
        points: raw.map((p, i) => ({ x: p.x, y: p.y, cluster: labels[i] < 0 ? -1 : labels[i] })),
        centroids: [],
      },
      metrics: {
        clusters: Math.max(-1, ...labels) + 1,
        noise: labels.filter((l) => l === -1).length,
      },
    });
  };

  snapshot(0);
  let cid = 0;
  for (let i = 0; i < n; i++) {
    if (labels[i] !== -2) continue;
    const nb = neighbors(i);
    if (nb.length < minPts) {
      labels[i] = -1; // 暂定噪声（后续可能被吸收为边界点）
      continue;
    }
    // 开新簇，BFS 扩展
    labels[i] = cid;
    const queue = [...nb];
    while (queue.length) {
      const j = queue.shift()!;
      if (labels[j] === -1) labels[j] = cid; // 噪声→边界点
      if (labels[j] !== -2) continue;
      labels[j] = cid;
      const nb2 = neighbors(j);
      if (nb2.length >= minPts) queue.push(...nb2);
    }
    cid++;
    snapshot(cid); // 每扩展完一个簇记一帧
  }
  snapshot(cid + 1); // 终态

  return {
    meta: {
      id: "dbscan",
      title: "DBSCAN 密度聚类",
      family: "clusters",
      algorithm: "DBSCAN",
      description: "3 个高斯团 + 均匀噪声。按密度可达性把点连成簇，孤立点判为噪声（灰色）。",
      tutorial: {
        problem: "同样是无标签聚类，但能发现任意形状的簇、识别噪声点，而且不用提前指定簇数。",
        intuition:
          "从一个“核心点”（周围足够密集）出发，把密度可达的点像滚雪球一样连成一簇；既不密集又连不上任何簇的孤立点，就判为噪声。",
        watch: [
          "簇一个个地长出来（每一帧多发现一个簇）",
          "灰色点是噪声，不属于任何簇",
          "全程不需要像 K-Means 那样预先告诉它有几个簇",
        ],
        concepts: [
          { term: "核心点", explain: "邻域(eps)内点数 ≥ minPts 的点，能发起扩展" },
          { term: "密度可达", explain: "通过一连串核心点能一路连过去" },
          { term: "噪声点", explain: "既不是核心点、也连不到任何簇的孤立点" },
        ],
        tryThis: "点“重新生成数据”，数一数它自动发现了几个簇、几个灰色噪声点。",
      },
      hyperparams: { eps, minPts, points: n },
    },
    frames,
  };
}
