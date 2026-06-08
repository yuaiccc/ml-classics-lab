import { Trajectory, Frame, ClustersState } from "@/player/types";

interface Point2D {
  x: number;
  y: number;
}

function generateData(): Point2D[] {
  const points: Point2D[] = [];
  const centers = [
    { cx: 2, cy: 3 },
    { cx: 6, cy: 7 },
    { cx: 7, cy: 2 },
  ];
  for (const c of centers) {
    for (let i = 0; i < 30; i++) {
      points.push({
        x: c.cx + (Math.random() - 0.5) * 2.5,
        y: c.cy + (Math.random() - 0.5) * 2.5,
      });
    }
  }
  return points;
}

function distance(a: Point2D, b: Point2D): number {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}

export function runKMeans(): Trajectory {
  const points = generateData();
  const k = 3;
  const maxIters = 30;

  let centroids: Point2D[] = [];
  for (let i = 0; i < k; i++) {
    centroids.push({
      x: Math.random() * 8 + 1,
      y: Math.random() * 8 + 1,
    });
  }

  const assignments = new Array(points.length).fill(0);

  const frames: Frame[] = [];

  const computeInertia = (): number => {
    let inertia = 0;
    for (let i = 0; i < points.length; i++) {
      const c = centroids[assignments[i]];
      inertia += distance(points[i], c) ** 2;
    }
    return inertia;
  };

  for (let iter = 0; iter <= maxIters; iter++) {
    const stateData: ClustersState = {
      points: points.map((p, i) => ({ x: p.x, y: p.y, cluster: assignments[i] })),
      centroids: centroids.map((c) => ({ x: c.x, y: c.y })),
      k,
    };

    frames.push({
      iter,
      state: { family: "clusters", data: stateData },
      metrics: { inertia: computeInertia() },
    });

    for (let i = 0; i < points.length; i++) {
      let minDist = Infinity;
      let minCluster = 0;
      for (let j = 0; j < k; j++) {
        const d = distance(points[i], centroids[j]);
        if (d < minDist) {
          minDist = d;
          minCluster = j;
        }
      }
      assignments[i] = minCluster;
    }

    const newCentroids: Point2D[] = [];
    for (let j = 0; j < k; j++) {
      const members = points.filter((_, i) => assignments[i] === j);
      if (members.length > 0) {
        newCentroids.push({
          x: members.reduce((s, p) => s + p.x, 0) / members.length,
          y: members.reduce((s, p) => s + p.y, 0) / members.length,
        });
      } else {
        newCentroids.push(centroids[j]);
      }
    }
    centroids = newCentroids;
  }

  return {
    meta: {
      id: "kmeans",
      title: "K-Means 聚类",
      algorithm: "K-Means",
      category: "unsupervised",
      source: "browser",
      abstract:
        "解决无标签数据的自动分组问题：给定一堆没有标签的数据点，如何自动将它们分成 K 个有意义的簇？K-Means 通过反复执行「分配→更新质心」两步，让质心逐帧移动到各簇的中心。这是无监督学习最经典的算法，也是理解聚类思想的起点。",
      description:
        "K-Means 从随机质心出发，交替执行分配和更新步骤，质心逐帧移向簇中心。30 次迭代后收敛。",
      hyperparams: {
        K: 3,
        max_iters: 30,
        samples: 90,
        init: "random",
      },
      insight:
        "K-Means 的两步迭代：①分配——每个点归入最近质心所在的簇；②更新——每个质心移到其簇内所有点的均值位置。保证收敛但不保证全局最优（依赖初始质心）。K 值需要人为指定，这是它的主要局限。",
    },
    frames,
  };
}
