// pca 家族：散点 + 均值 + 当前主成分方向（含上一帧虚线，显示旋转收敛）。
import { PCAState } from "@/player/types";
import { makeScale } from "./plot";

const W = 560;
const H = 380;
const M = 28;
const LEN = 6; // 主轴半长（数据坐标）

export default function PCAPlot({ state }: { state: PCAState }) {
  const { points, mean, axis, prevAxis } = state;
  const s = makeScale(points, W, H, M);

  const axisLine = (v: { x: number; y: number }) => ({
    x1: s.sx(mean.x - v.x * LEN),
    y1: s.sy(mean.y - v.y * LEN),
    x2: s.sx(mean.x + v.x * LEN),
    y2: s.sy(mean.y + v.y * LEN),
  });
  const cur = axisLine(axis);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full rounded-xl bg-[rgba(10,14,23,0.6)]">
      <rect
        x={M}
        y={M}
        width={W - 2 * M}
        height={H - 2 * M}
        fill="none"
        stroke="rgba(0,255,136,0.12)"
      />

      {/* 数据点 */}
      {points.map((p, i) => (
        <circle key={i} cx={s.sx(p.x)} cy={s.sy(p.y)} r={3.5} fill="#00e5ff" opacity={0.6} />
      ))}

      {/* 上一帧主轴（淡） */}
      {prevAxis && (() => {
        const pv = axisLine(prevAxis);
        return (
          <line
            x1={pv.x1}
            y1={pv.y1}
            x2={pv.x2}
            y2={pv.y2}
            stroke="#b388ff"
            strokeWidth={1.5}
            strokeDasharray="4 4"
            opacity={0.4}
          />
        );
      })()}

      {/* 当前主成分方向 */}
      <line x1={cur.x1} y1={cur.y1} x2={cur.x2} y2={cur.y2} stroke="#00ff88" strokeWidth={3} />

      {/* 均值点 */}
      <circle cx={s.sx(mean.x)} cy={s.sy(mean.y)} r={5} fill="#ffab40" stroke="#0a0e17" strokeWidth={1.5} />

      <text x={M + 8} y={M + 18} className="font-mono" fill="#00ff88" fontSize={12}>
        PC1 ≈ ({axis.x.toFixed(2)}, {axis.y.toFixed(2)})
      </text>
    </svg>
  );
}
