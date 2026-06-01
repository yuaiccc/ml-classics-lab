// 二维二分类玩具数据集，供分类/聚类算法共用。
import { LabeledPoint } from "@/player/types";
import { mulberry32, gaussian } from "./rng";

/** 两个高斯团簇，线性可分（用于感知机 / 逻辑回归） */
export function makeBlobs(seed: number, nPerClass = 40, sep = 3.2): LabeledPoint[] {
  const rng = mulberry32(seed);
  const pts: LabeledPoint[] = [];
  const centers = [
    { x: -sep / 2, y: -sep / 2 },
    { x: sep / 2, y: sep / 2 },
  ];
  centers.forEach((c, label) => {
    for (let i = 0; i < nPerClass; i++) {
      pts.push({ x: c.x + gaussian(rng, 0, 1), y: c.y + gaussian(rng, 0, 1), label });
    }
  });
  return pts;
}

/** 同心圆：内圈类 0、外环类 1，线性不可分（用于决策树 / KNN 展示非线性边界） */
export function makeCircles(seed: number, nPerClass = 60): LabeledPoint[] {
  const rng = mulberry32(seed);
  const pts: LabeledPoint[] = [];
  for (let i = 0; i < nPerClass; i++) {
    const a = rng() * Math.PI * 2;
    const r = gaussian(rng, 1.2, 0.35);
    pts.push({ x: r * Math.cos(a), y: r * Math.sin(a), label: 0 });
  }
  for (let i = 0; i < nPerClass; i++) {
    const a = rng() * Math.PI * 2;
    const r = gaussian(rng, 3.4, 0.4);
    pts.push({ x: r * Math.cos(a), y: r * Math.sin(a), label: 1 });
  }
  return pts;
}

/** 一团各向异性的点云，主方向明显（用于 PCA） */
export function makeAnisotropic(seed: number, n = 160): { x: number; y: number }[] {
  const rng = mulberry32(seed);
  const angle = Math.PI / 5;
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  const pts: { x: number; y: number }[] = [];
  for (let i = 0; i < n; i++) {
    const u = gaussian(rng, 0, 3.2); // 主方向方差大
    const v = gaussian(rng, 0, 0.7); // 次方向方差小
    pts.push({ x: u * cos - v * sin, y: u * sin + v * cos });
  }
  return pts;
}
