// gridworld 家族：价值热图 + 贪心策略箭头 + 目标/陷阱标记。
import { GridWorldState } from "@/player/types";

const CELL = 56;

const ARROW = ["↑", "→", "↓", "←"];

export default function GridWorldPlot({ state }: { state: GridWorldState }) {
  const { rows, cols, values, policy, goal, pit } = state;
  const W = cols * CELL;
  const H = rows * CELL;
  const vmax = Math.max(0.01, ...values.map(Math.abs));

  const color = (v: number) => {
    const t = Math.min(1, Math.abs(v) / vmax);
    return v >= 0 ? `rgba(0,255,136,${0.05 + 0.7 * t})` : `rgba(255,82,82,${0.05 + 0.7 * t})`;
  };

  return (
    <div className="rounded-xl bg-[rgba(10,14,23,0.6)] p-5 flex justify-center">
      <svg width={W} height={H} className="rounded-lg">
        {values.map((v, s) => {
          const r = Math.floor(s / cols);
          const c = s % cols;
          const x = c * CELL;
          const y = r * CELL;
          const isGoal = s === goal;
          const isPit = s === pit;
          return (
            <g key={s}>
              <rect x={x + 1} y={y + 1} width={CELL - 2} height={CELL - 2} rx={4} fill={color(v)} stroke="rgba(148,163,184,0.15)" />
              {isGoal && <text x={x + CELL / 2} y={y + CELL / 2 + 7} textAnchor="middle" fontSize={22}>🏁</text>}
              {isPit && <text x={x + CELL / 2} y={y + CELL / 2 + 7} textAnchor="middle" fontSize={20}>⛔</text>}
              {!isGoal && !isPit && (
                <>
                  <text x={x + CELL / 2} y={y + CELL / 2 + 2} textAnchor="middle" fill="#e2e8f0" fontSize={18} opacity={0.85}>
                    {ARROW[policy[s]] ?? ""}
                  </text>
                  <text x={x + CELL / 2} y={y + CELL - 6} textAnchor="middle" fill="#64748b" fontSize={9} className="font-mono">
                    {v.toFixed(2)}
                  </text>
                </>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
