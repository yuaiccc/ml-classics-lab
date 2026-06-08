import { Trajectory, Frame, CurvesState } from "@/player/types";

interface Point3D {
  x: number;
  y: number;
  z: number;
  label: number;
}

function generateData(): Point3D[] {
  const points: Point3D[] = [];
  const centers = [
    { cx: 2, cy: 3, cz: 1, label: 0 },
    { cx: 6, cy: 7, cz: 5, label: 1 },
    { cx: 7, cy: 2, cz: 8, label: 2 },
  ];
  for (const c of centers) {
    for (let i = 0; i < 30; i++) {
      points.push({
        x: c.cx + (Math.random() - 0.5) * 2,
        y: c.cy + (Math.random() - 0.5) * 2,
        z: c.cz + (Math.random() - 0.5) * 2,
        label: c.label,
      });
    }
  }
  return points;
}

function mean(arr: number[]): number {
  return arr.reduce((s, v) => s + v, 0) / arr.length;
}

function dot(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function powerIteration(cov: number[][], nIter: number): number[] {
  const n = cov.length;
  let v = Array.from({ length: n }, () => Math.random() - 0.5);
  let norm = Math.sqrt(dot(v, v));
  v = v.map((x) => x / norm);

  for (let it = 0; it < nIter; it++) {
    const newV: number[] = [];
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = 0; j < n; j++) s += cov[i][j] * v[j];
      newV.push(s);
    }
    norm = Math.sqrt(dot(newV, newV));
    v = newV.map((x) => x / norm);
  }

  return v;
}

function deflate(cov: number[][], eigenvec: number[]): number[][] {
  const n = cov.length;
  const lambda = dot(eigenvec, cov.map((row) => dot(row, eigenvec)));
  const result: number[][] = [];
  for (let i = 0; i < n; i++) {
    result[i] = [];
    for (let j = 0; j < n; j++) {
      result[i][j] = cov[i][j] - lambda * eigenvec[i] * eigenvec[j];
    }
  }
  return result;
}

export function runPCA(): Trajectory {
  const points = generateData();
  const n = points.length;
  const dims = 3;

  const mx = mean(points.map((p) => p.x));
  const my = mean(points.map((p) => p.y));
  const mz = mean(points.map((p) => p.z));

  const centered = points.map((p) => [p.x - mx, p.y - my, p.z - mz]);

  const cov: number[][] = Array.from({ length: dims }, () => Array(dims).fill(0));
  for (let i = 0; i < dims; i++) {
    for (let j = i; j < dims; j++) {
      let s = 0;
      for (const c of centered) s += c[i] * c[j];
      cov[i][j] = s / (n - 1);
      cov[j][i] = cov[i][j];
    }
  }

  const totalVariance = cov[0][0] + cov[1][1] + cov[2][2];

  const pc1 = powerIteration(cov, 200);
  const cov2 = deflate(cov, pc1);
  const pc2 = powerIteration(cov2, 200);

  const lambda1 = dot(pc1, cov.map((row) => dot(row, pc1)));
  const lambda2 = dot(pc2, cov2.map((row) => dot(row, pc2)));

  const projections = centered.map((c) => ({
    x: dot(c, pc1),
    y: dot(c, pc2),
    label: points[centered.indexOf(c)].label,
  }));

  const totalFrames = 60;
  const frames: Frame[] = [];

  for (let iter = 0; iter <= totalFrames; iter++) {
    const t = iter / totalFrames;
    const eased = t < 1 ? 1 - Math.pow(1 - t, 3) : 1;

    const embedding = projections.map((p) => ({
      x: p.x * eased,
      y: p.y * eased,
      label: p.label,
    }));

    const explainedVar = (lambda1 + lambda2) / totalVariance * eased;

    const stateData: CurvesState = { embedding };

    frames.push({
      iter,
      state: { family: "curves", data: stateData },
      metrics: {
        explained_variance: explainedVar,
        pc1_variance: (lambda1 / totalVariance) * eased,
        pc2_variance: (lambda2 / totalVariance) * eased,
      },
    });
  }

  return {
    meta: {
      id: "pca",
      title: "PCA · 主成分分析",
      algorithm: "PCA",
      category: "unsupervised",
      source: "browser",
      abstract:
        "解决高维数据降维问题：3D 数据如何投影到 2D 平面同时保留最多的信息？PCA 通过寻找方差最大的方向（主成分），将数据从 3 维降到 2 维，动画展示数据点从原始空间逐步展开到主成分平面的过程。",
      description:
        "PCA 用幂迭代法求协方差矩阵的特征向量，将 3D 数据投影到前两个主成分方向。",
      hyperparams: {
        dimensions: 3,
        target_dims: 2,
        samples: 90,
        method: "power_iteration",
      },
      insight:
        "PCA 的核心：找方差最大的方向作为主成分，本质是协方差矩阵的特征分解。第一主成分捕获最大方差，第二主成分捕获正交方向上的次大方差。降维后保留的方差比例（explained variance）衡量信息保留程度。",
    },
    frames,
  };
}
