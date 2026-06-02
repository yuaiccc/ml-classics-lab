// multiclass 家族：多类决策边界（背景按预测类别着色）+ 真实数据点 + 坐标轴/图例。
import { MultiBoundaryState } from "@/player/types";
import { CLUSTER_COLORS } from "./plot";

const W = 560;
const H = 400;
const M = 44;

export default function MultiBoundaryPlot({ state }: { state: MultiBoundaryState }) {
  const { points, grid, classNames, xName, yName } = state;
  const gx = (x: number) => M + ((x - grid.x0) / (grid.x1 - grid.x0)) * (W - 2 * M);
  const gy = (y: number) => H - M - ((y - grid.y0) / (grid.y1 - grid.y0)) * (H - 2 * M);
  const cellW = (W - 2 * M) / grid.cols + 0.5;
  const cellH = (H - 2 * M) / grid.rows + 0.5;
  const dx = (grid.x1 - grid.x0) / grid.cols;
  const dy = (grid.y1 - grid.y0) / grid.rows;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full rounded-xl bg-[rgba(10,14,23,0.6)]">
      {/* 决策区背景 */}
      {grid.values.map((cls, idx) => {
        const c = idx % grid.cols;
        const r = Math.floor(idx / grid.cols);
        const x = grid.x0 + c * dx;
        const yTop = grid.y0 + (r + 1) * dy;
        return (
          <rect
            key={idx}
            x={gx(x)}
            y={gy(yTop)}
            width={cellW}
            height={cellH}
            fill={CLUSTER_COLORS[cls % CLUSTER_COLORS.length]}
            fillOpacity={0.16}
          />
        );
      })}

      {/* 数据点 */}
      {points.map((p, i) => (
        <circle key={i} cx={gx(p.x)} cy={gy(p.y)} r={4} fill={CLUSTER_COLORS[p.label % CLUSTER_COLORS.length]} stroke="rgba(255,255,255,0.55)" strokeWidth={0.8} />
      ))}

      <rect x={M} y={M} width={W - 2 * M} height={H - 2 * M} fill="none" stroke="rgba(0,255,136,0.12)" />

      {/* 坐标轴名 */}
      <text x={W / 2} y={H - 12} textAnchor="middle" fill="#64748b" fontSize={12}>{xName}</text>
      <text x={14} y={H / 2} textAnchor="middle" fill="#64748b" fontSize={12} transform={`rotate(-90 14 ${H / 2})`}>{yName}</text>

      {/* 图例 */}
      {classNames.map((name, i) => (
        <g key={i} transform={`translate(${M + 8}, ${M + 8 + i * 18})`}>
          <circle cx={0} cy={-4} r={5} fill={CLUSTER_COLORS[i % CLUSTER_COLORS.length]} />
          <text x={10} y={0} fill="#cbd5e1" fontSize={12} className="font-mono">{name}</text>
        </g>
      ))}
    </svg>
  );
}
