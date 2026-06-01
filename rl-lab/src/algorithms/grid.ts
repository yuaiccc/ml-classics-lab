// 在给定矩形区域上采样一个 cols×rows 网格，对每个格点调用 score 函数，
// 得到决策边界网格（见 BoundaryState）。可视化器据此着色。
import { BoundaryGrid, LabeledPoint } from "@/player/types";

export function computeBounds(points: { x: number; y: number }[], pad = 0.6) {
  let x0 = Infinity;
  let x1 = -Infinity;
  let y0 = Infinity;
  let y1 = -Infinity;
  for (const p of points) {
    if (p.x < x0) x0 = p.x;
    if (p.x > x1) x1 = p.x;
    if (p.y < y0) y0 = p.y;
    if (p.y > y1) y1 = p.y;
  }
  const px = (x1 - x0) * pad || 1;
  const py = (y1 - y0) * pad || 1;
  return { x0: x0 - px, x1: x1 + px, y0: y0 - py, y1: y1 + py };
}

/** score(x,y) 返回类别 1 的得分/概率 ∈ [0,1] */
export function makeGrid(
  bounds: { x0: number; y0: number; x1: number; y1: number },
  cols: number,
  rows: number,
  score: (x: number, y: number) => number
): BoundaryGrid {
  const values = new Array(cols * rows);
  for (let r = 0; r < rows; r++) {
    const y = bounds.y0 + ((r + 0.5) / rows) * (bounds.y1 - bounds.y0);
    for (let c = 0; c < cols; c++) {
      const x = bounds.x0 + ((c + 0.5) / cols) * (bounds.x1 - bounds.x0);
      values[r * cols + c] = score(x, y);
    }
  }
  return { ...bounds, cols, rows, values };
}

export const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

/** 训练集统一标准化所需的 bounds（边界算法共用一个固定区域，避免逐帧抖动） */
export function fixedBounds(points: LabeledPoint[]) {
  return computeBounds(points);
}
