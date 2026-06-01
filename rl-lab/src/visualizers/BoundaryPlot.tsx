// classification 家族：决策边界网格着色 + 带标签的点 + 可选线性边界。
import { BoundaryState } from "@/player/types";

const W = 560;
const H = 380;
const M = 20;

const C1 = "#00e5ff"; // 类 1
const C0 = "#ffab40"; // 类 0

export default function BoundaryPlot({ state }: { state: BoundaryState }) {
  const { points, grid, line } = state;
  const gx = (x: number) => M + ((x - grid.x0) / (grid.x1 - grid.x0)) * (W - 2 * M);
  const gy = (y: number) => H - M - ((y - grid.y0) / (grid.y1 - grid.y0)) * (H - 2 * M);

  const cellW = (W - 2 * M) / grid.cols + 0.5;
  const cellH = (H - 2 * M) / grid.rows + 0.5;
  const dx = (grid.x1 - grid.x0) / grid.cols;
  const dy = (grid.y1 - grid.y0) / grid.rows;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full rounded-xl bg-[rgba(10,14,23,0.6)]">
      <defs>
        <clipPath id="bnd-area">
          <rect x={M} y={M} width={W - 2 * M} height={H - 2 * M} />
        </clipPath>
      </defs>

      {/* 决策边界网格 */}
      <g>
        {grid.values.map((v, idx) => {
          const c = idx % grid.cols;
          const r = Math.floor(idx / grid.cols);
          const x = grid.x0 + c * dx;
          const yTop = grid.y0 + (r + 1) * dy;
          const conf = Math.abs(v - 0.5) * 2; // 0..1
          return (
            <rect
              key={idx}
              x={gx(x)}
              y={gy(yTop)}
              width={cellW}
              height={cellH}
              fill={v >= 0.5 ? C1 : C0}
              fillOpacity={0.08 + 0.32 * conf}
            />
          );
        })}
      </g>

      {/* 决策边界线（线性分类器） */}
      {line && (
        <line
          x1={gx(grid.x0)}
          y1={gy(line.slope * grid.x0 + line.intercept)}
          x2={gx(grid.x1)}
          y2={gy(line.slope * grid.x1 + line.intercept)}
          stroke="#fff"
          strokeWidth={2}
          strokeDasharray="5 3"
          clipPath="url(#bnd-area)"
        />
      )}

      {/* 数据点 */}
      {points.map((p, i) => (
        <circle
          key={i}
          cx={gx(p.x)}
          cy={gy(p.y)}
          r={4}
          fill={p.label === 1 ? C1 : C0}
          stroke="rgba(255,255,255,0.6)"
          strokeWidth={0.8}
        />
      ))}

      {/* 边框 */}
      <rect
        x={M}
        y={M}
        width={W - 2 * M}
        height={H - 2 * M}
        fill="none"
        stroke="rgba(0,255,136,0.12)"
      />
    </svg>
  );
}
