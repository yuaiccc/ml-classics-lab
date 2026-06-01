// 极简坐标映射：把数据域映射到 SVG 像素，自动留 10% 边距。
// 暂不引 d3，等坐标轴需求复杂了再说（见 ML_LAB_DESIGN.md §7 开放问题）。
import { Point2D } from "@/player/types";

export interface Scale {
  sx: (x: number) => number;
  sy: (y: number) => number;
}

/** 用 pts（+可选 extra）确定数据范围，返回到 [margin, size-margin] 的映射 */
export function makeScale(
  pts: Point2D[],
  width: number,
  height: number,
  margin: number,
  extra: Point2D[] = []
): Scale {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const p of [...pts, ...extra]) {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }
  if (!isFinite(minX)) {
    minX = -1;
    maxX = 1;
    minY = -1;
    maxY = 1;
  }
  const padX = (maxX - minX) * 0.1 || 1;
  const padY = (maxY - minY) * 0.1 || 1;
  minX -= padX;
  maxX += padX;
  minY -= padY;
  maxY += padY;

  const sx = (x: number) => margin + ((x - minX) / (maxX - minX)) * (width - 2 * margin);
  const sy = (y: number) =>
    height - margin - ((y - minY) / (maxY - minY)) * (height - 2 * margin);
  return { sx, sy };
}

/** 簇配色，复用项目霓虹色板 */
export const CLUSTER_COLORS = [
  "#00ff88",
  "#00e5ff",
  "#b388ff",
  "#ffab40",
  "#ff5252",
  "#7c4dff",
  "#64ffda",
  "#ff80ab",
];
