// curvefit 家族：真实函数（灰虚线）+ 训练点（青）+ 当前拟合曲线（绿）。
import { CurveFitState } from "@/player/types";
import { makeScale } from "./plot";

const W = 560;
const H = 380;
const M = 30;
// 固定视野，避免拟合曲线抖动导致坐标乱跳
const CORNERS = [
  { x: -1.35, y: -2.4 },
  { x: 1.35, y: 2.4 },
];

export default function CurveFitPlot({ state }: { state: CurveFitState }) {
  const { train, fit, truth, caption } = state;
  const s = makeScale(CORNERS, W, H, M);
  const path = (pts: { x: number; y: number }[]) =>
    pts.map((p, i) => `${i === 0 ? "M" : "L"}${s.sx(p.x).toFixed(1)},${s.sy(p.y).toFixed(1)}`).join(" ");

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full rounded-xl bg-[rgba(10,14,23,0.6)]">
      <defs>
        <clipPath id="cf-area">
          <rect x={M} y={M} width={W - 2 * M} height={H - 2 * M} />
        </clipPath>
      </defs>
      <rect x={M} y={M} width={W - 2 * M} height={H - 2 * M} fill="none" stroke="rgba(0,255,136,0.12)" />

      <g clipPath="url(#cf-area)">
        {/* 真实函数 */}
        <path d={path(truth)} fill="none" stroke="#64748b" strokeWidth={2} strokeDasharray="5 4" />
        {/* 当前拟合 */}
        <path d={path(fit)} fill="none" stroke="#00ff88" strokeWidth={2.5} />
        {/* 训练点 */}
        {train.map((p, i) => (
          <circle key={i} cx={s.sx(p.x)} cy={s.sy(p.y)} r={4} fill="#00e5ff" stroke="rgba(255,255,255,0.5)" strokeWidth={0.8} />
        ))}
      </g>

      <text x={M + 8} y={M + 18} className="font-mono" fill="#00ff88" fontSize={13}>{caption}</text>
      {/* 图例 */}
      <g transform={`translate(${W - 150}, ${M + 12})`} fontSize={11} className="font-mono">
        <line x1={0} y1={-4} x2={18} y2={-4} stroke="#64748b" strokeWidth={2} strokeDasharray="4 3" />
        <text x={24} y={0} fill="#94a3b8">真实函数</text>
        <line x1={0} y1={12} x2={18} y2={12} stroke="#00ff88" strokeWidth={2.5} />
        <text x={24} y={16} fill="#94a3b8">当前拟合</text>
      </g>
    </svg>
  );
}
