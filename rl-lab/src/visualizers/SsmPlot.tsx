// ssm 家族（Mamba 选择性扫描）：序列 + 选择门 + 记忆热图，按扫描位置逐列点亮。
import { SSMState } from "@/player/types";

const letter = (id: number) => (id === 0 ? "·" : String.fromCharCode(64 + id)); // 0=· ,1=A..
const COL = 34;
const LEFT = 72;

export default function SsmPlot({ state }: { state: SSMState }) {
  const { tokens, pos, gates, memory, output } = state;
  const L = tokens.length;
  const C = memory[0]?.length ?? 4;

  const tokenY = 18;
  const gateTop = 44;
  const gateH = 52;
  const gateBase = gateTop + gateH;
  const memTop = gateBase + 24;
  const memCell = 22;
  const W = LEFT + L * COL + 12;
  const H = memTop + C * memCell + 26;

  const colDim = (c: number) => (c > pos ? 0.18 : 1); // 未扫描到的列变暗

  return (
    <div className="rounded-xl bg-[rgba(10,14,23,0.6)] p-4">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
        {/* 左侧行标签 */}
        <text x={LEFT - 10} y={tokenY + 4} textAnchor="end" fill="#64748b" fontSize={12}>序列</text>
        <text x={LEFT - 10} y={gateTop + gateH / 2} textAnchor="end" fill="#64748b" fontSize={12}>选择门 Δ</text>
        <text x={LEFT - 10} y={memTop - 8} textAnchor="end" fill="#64748b" fontSize={12}>记忆</text>

        {/* 当前扫描列高亮背景 */}
        <rect x={LEFT + pos * COL - 1} y={tokenY - 16} width={COL} height={H - tokenY} fill="rgba(179,136,255,0.12)" rx={4} />

        {tokens.map((tok, c) => {
          const x = LEFT + c * COL;
          const cx = x + COL / 2;
          return (
            <g key={c} opacity={colDim(c)}>
              {/* token */}
              <rect
                x={x + 2}
                y={tokenY - 14}
                width={COL - 6}
                height={26}
                rx={4}
                fill={tok > 0 ? "rgba(0,229,255,0.18)" : "rgba(100,116,139,0.12)"}
                stroke={tok > 0 ? "rgba(0,229,255,0.5)" : "transparent"}
              />
              <text x={cx} y={tokenY + 4} textAnchor="middle" fill={tok > 0 ? "#00e5ff" : "#475569"} fontSize={13} className="font-mono">
                {letter(tok)}
              </text>

              {/* 选择门柱 */}
              <rect x={cx - 7} y={gateBase - gates[c] * gateH} width={14} height={Math.max(1, gates[c] * gateH)} rx={2} fill="#00ff88" opacity={0.4 + 0.6 * gates[c]} />

              {/* 记忆列（C 个通道） */}
              {memory[c].map((v, ch) => (
                <rect
                  key={ch}
                  x={x + 2}
                  y={memTop + ch * memCell}
                  width={COL - 6}
                  height={memCell - 2}
                  rx={2}
                  fill={`rgba(179,136,255,${0.06 + 0.94 * Math.min(1, v)})`}
                />
              ))}
            </g>
          );
        })}

        {/* 记忆通道行标签 A..D */}
        {Array.from({ length: C }, (_, ch) => (
          <text key={ch} x={LEFT - 10} y={memTop + ch * memCell + memCell / 2 + 2} textAnchor="end" fill="#475569" fontSize={11} className="font-mono">
            {String.fromCharCode(65 + ch)}
          </text>
        ))}
      </svg>

      <div className="mt-2 text-xs text-slate-500 font-mono">
        扫描到第 <span className="text-[#b388ff]">{pos + 1}</span>/{L} 位 · 当前记忆里的信号：
        <span className="text-[#00ff88] ml-1">{output[pos] ? letter(output[pos]) : "（空）"}</span>
      </div>
    </div>
  );
}
