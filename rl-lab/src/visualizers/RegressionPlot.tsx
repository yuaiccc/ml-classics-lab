// scatter-boundary 家族（回归变体）：散点 + 当前拟合直线。
import { RegressionState } from "@/player/types";
import { makeScale } from "./plot";

const W = 560;
const H = 360;
const M = 36;

export default function RegressionPlot({ state }: { state: RegressionState }) {
  const { points, fit } = state;
  const s = makeScale(points, W, H, M);

  const xs = points.map((p) => p.x);
  const xmin = Math.min(...xs);
  const xmax = Math.max(...xs);
  const a = { x: xmin, y: fit.slope * xmin + fit.intercept };
  const b = { x: xmax, y: fit.slope * xmax + fit.intercept };

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full rounded-xl bg-[rgba(10,14,23,0.6)]">
      <defs>
        <clipPath id="plot-area">
          <rect x={M} y={M} width={W - 2 * M} height={H - 2 * M} />
        </clipPath>
      </defs>

      {/* 边框 */}
      <rect
        x={M}
        y={M}
        width={W - 2 * M}
        height={H - 2 * M}
        fill="none"
        stroke="rgba(0,255,136,0.12)"
      />

      {/* 数据点 */}
      <g clipPath="url(#plot-area)">
        {points.map((p, i) => (
          <circle key={i} cx={s.sx(p.x)} cy={s.sy(p.y)} r={3.5} fill="#00e5ff" opacity={0.7} />
        ))}

        {/* 当前拟合直线 */}
        <line
          x1={s.sx(a.x)}
          y1={s.sy(a.y)}
          x2={s.sx(b.x)}
          y2={s.sy(b.y)}
          stroke="#00ff88"
          strokeWidth={2.5}
        />
      </g>

      {/* 方程 */}
      <text x={M + 8} y={M + 18} className="font-mono" fill="#00ff88" fontSize={13}>
        y = {fit.slope.toFixed(3)}x {fit.intercept >= 0 ? "+" : "−"}{" "}
        {Math.abs(fit.intercept).toFixed(3)}
      </text>
    </svg>
  );
}
