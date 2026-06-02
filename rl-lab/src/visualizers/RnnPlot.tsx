// rnn 家族：输入比特序列 + 隐藏状态随时间的热图 + 目标/预测。
import { RnnState } from "@/player/types";

const COL = 40;
const LEFT = 60;

export default function RnnPlot({ state }: { state: RnnState }) {
  const { inputs, hidden, target, pred } = state;
  const T = inputs.length;
  const H = hidden[0]?.length ?? 0;
  const cell = 18;
  const W = LEFT + T * COL + 12;
  const top = 30;
  const memTop = top + 40;
  const Hh = memTop + H * cell + 30;

  return (
    <div className="rounded-xl bg-[rgba(10,14,23,0.6)] p-5">
      <svg viewBox={`0 0 ${W} ${Hh}`} className="w-full">
        <text x={LEFT - 10} y={top + 4} textAnchor="end" fill="#64748b" fontSize={12}>输入</text>
        <text x={LEFT - 10} y={memTop - 8} textAnchor="end" fill="#64748b" fontSize={12}>隐藏状态 h</text>

        {/* 输入比特 */}
        {inputs.map((b, t) => (
          <g key={t}>
            <rect x={LEFT + t * COL + 4} y={top - 14} width={COL - 8} height={26} rx={4}
              fill={b ? "rgba(0,229,255,0.2)" : "rgba(100,116,139,0.12)"} stroke={b ? "rgba(0,229,255,0.5)" : "transparent"} />
            <text x={LEFT + t * COL + COL / 2} y={top + 4} textAnchor="middle" fill={b ? "#00e5ff" : "#475569"} fontSize={14} className="font-mono">{b}</text>
          </g>
        ))}

        {/* 隐藏状态热图：行=维度，列=时间 */}
        {hidden.map((h, t) =>
          h.map((v, j) => (
            <rect key={`${t}-${j}`} x={LEFT + t * COL + 4} y={memTop + j * cell} width={COL - 8} height={cell - 2} rx={2}
              fill={v >= 0 ? `rgba(0,255,136,${Math.min(1, Math.abs(v))})` : `rgba(179,136,255,${Math.min(1, Math.abs(v))})`} />
          ))
        )}

        <text x={LEFT} y={Hh - 8} fill="#64748b" fontSize={12} className="font-mono">
          目标奇偶: <tspan fill="#00e5ff">{target}</tspan>　预测:{" "}
          <tspan fill={pred === target ? "#00ff88" : "#ff5252"}>{pred}</tspan>
          <tspan fill="#475569">　（绿=正激活 / 紫=负激活）</tspan>
        </text>
      </svg>
    </div>
  );
}
