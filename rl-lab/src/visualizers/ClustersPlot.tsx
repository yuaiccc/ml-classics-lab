// clusters 家族：点按当前归属着色 + 质心（叉号）。
import { ClusterState } from "@/player/types";
import { makeScale, CLUSTER_COLORS } from "./plot";

const W = 560;
const H = 360;
const M = 24;

export default function ClustersPlot({ state }: { state: ClusterState }) {
  const { points, centroids } = state;
  const s = makeScale(points, W, H, M, centroids);

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
        <circle
          key={i}
          cx={s.sx(p.x)}
          cy={s.sy(p.y)}
          r={3.5}
          fill={p.cluster < 0 ? "#475569" : CLUSTER_COLORS[p.cluster % CLUSTER_COLORS.length]}
          opacity={p.cluster < 0 ? 0.5 : 0.65}
        />
      ))}

      {/* 质心：描黑边的叉号，醒目 */}
      {centroids.map((c, i) => {
        const cx = s.sx(c.x);
        const cy = s.sy(c.y);
        const col = CLUSTER_COLORS[i % CLUSTER_COLORS.length];
        const r = 9;
        return (
          <g key={i} stroke={col} strokeWidth={3} strokeLinecap="round">
            <line x1={cx - r} y1={cy - r} x2={cx + r} y2={cy + r} />
            <line x1={cx - r} y1={cy + r} x2={cx + r} y2={cy - r} />
            <circle cx={cx} cy={cy} r={r + 4} fill="none" stroke={col} strokeWidth={1.5} opacity={0.5} />
          </g>
        );
      })}
    </svg>
  );
}
