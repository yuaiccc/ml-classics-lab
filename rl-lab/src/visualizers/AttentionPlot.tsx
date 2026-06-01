// attention 家族：self-attention 权重热图（行=输出位置，列=输入位置）。
// 训练中从噪声收敛到反对角线 = 学会“反转”这个信息路由。
import { AttentionState } from "@/player/types";

const letter = (id: number) => String.fromCharCode(65 + id); // 0->A, 1->B ...
const CELL = 44;
const PAD = 30;

function Chip({ id, ok }: { id: number; ok?: boolean }) {
  return (
    <span
      className={`inline-flex items-center justify-center w-8 h-8 rounded-md font-mono text-sm border ${
        ok === undefined
          ? "border-slate-700 text-slate-300 bg-[rgba(0,229,255,0.08)]"
          : ok
          ? "border-[#00ff88]/40 text-[#00ff88] bg-[rgba(0,255,136,0.1)]"
          : "border-[#ff5252]/40 text-[#ff5252] bg-[rgba(255,82,82,0.1)]"
      }`}
    >
      {letter(id)}
    </span>
  );
}

export default function AttentionPlot({ state }: { state: AttentionState }) {
  const { tokens, target, pred, attention } = state;
  const L = tokens.length;
  const W = PAD + L * CELL + 10;
  const H = PAD + L * CELL + 10;

  return (
    <div className="rounded-xl bg-[rgba(10,14,23,0.6)] p-5 flex flex-col gap-4">
      <div className="flex flex-wrap gap-6 items-start">
        {/* 注意力热图 */}
        <svg width={W} height={H} className="shrink-0">
          {/* 列标签：输入 token（键位） */}
          {tokens.map((t, c) => (
            <text
              key={`col-${c}`}
              x={PAD + c * CELL + CELL / 2}
              y={PAD - 12}
              textAnchor="middle"
              fill="#64748b"
              fontSize={13}
              className="font-mono"
            >
              {letter(t)}
            </text>
          ))}
          {/* 行标签：输出位置（查询位） */}
          {tokens.map((_, r) => (
            <text
              key={`row-${r}`}
              x={PAD - 10}
              y={PAD + r * CELL + CELL / 2 + 4}
              textAnchor="end"
              fill="#475569"
              fontSize={12}
              className="font-mono"
            >
              {r + 1}
            </text>
          ))}
          {/* 单元格 */}
          {attention.map((row, r) => {
            const maxJ = row.indexOf(Math.max(...row));
            return row.map((w, c) => (
              <g key={`${r}-${c}`}>
                <rect
                  x={PAD + c * CELL}
                  y={PAD + r * CELL}
                  width={CELL - 2}
                  height={CELL - 2}
                  rx={3}
                  fill={`rgba(0,255,136,${0.05 + 0.95 * w})`}
                  stroke={c === maxJ ? "rgba(0,255,136,0.7)" : "transparent"}
                  strokeWidth={1.5}
                />
                <text
                  x={PAD + c * CELL + (CELL - 2) / 2}
                  y={PAD + r * CELL + (CELL - 2) / 2 + 4}
                  textAnchor="middle"
                  fill={w > 0.5 ? "#0a0e17" : "#64748b"}
                  fontSize={10}
                  className="font-mono"
                >
                  {w.toFixed(2)}
                </text>
              </g>
            ));
          })}
        </svg>

        <div className="text-xs text-slate-500 leading-relaxed max-w-[200px] pt-6">
          <div className="text-slate-300 font-semibold mb-1">怎么读这张图</div>
          行 = 第几个<span className="text-slate-300">输出</span>位置；列 = 它关注的
          <span className="text-slate-300">输入</span> token。亮格 = 关注强。
          学成后是一条<span className="text-[#00ff88]">反对角线</span>：输出第1个关注输入最后一个 → 实现反转。
        </div>
      </div>

      {/* 输入 / 目标 / 预测 序列 */}
      <div className="flex flex-col gap-2 text-sm">
        <div className="flex items-center gap-2">
          <span className="text-slate-500 w-12 text-xs">输入</span>
          {tokens.map((t, i) => (
            <Chip key={i} id={t} />
          ))}
        </div>
        <div className="flex items-center gap-2">
          <span className="text-slate-500 w-12 text-xs">预测</span>
          {pred.map((t, i) => (
            <Chip key={i} id={t} ok={t === target[i]} />
          ))}
          <span className="text-xs text-slate-600 ml-2">（目标：反转）</span>
        </div>
      </div>
    </div>
  );
}
