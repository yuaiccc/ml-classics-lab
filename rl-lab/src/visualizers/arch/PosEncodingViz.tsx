import { useEffect, useMemo, useRef, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from "recharts";
import { PosEncState, PosEncVariant } from "./types";
import { sinusoidalPE, ropeApply, ropeAngle, yarnAngle } from "@/algorithms/arch/pos-encoding-fns";

const VARIANTS: { key: PosEncVariant; label: string }[] = [
  { key: "sinusoidal", label: "正弦 PE" },
  { key: "rope", label: "RoPE" },
  { key: "yarn", label: "YaRN" },
  { key: "nope", label: "NoPE" },
];

function Heatmap({ maxPos, dim, curPos }: { maxPos: number; dim: number; curPos: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const cell = 6;
  useEffect(() => {
    const ctx = ref.current?.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, dim * cell, maxPos * cell);
    for (let p = 0; p < maxPos; p++) {
      const pe = sinusoidalPE(p, dim);
      for (let i = 0; i < dim; i++) {
        const t = (pe[i] + 1) / 2; // [-1,1]→[0,1]
        const r = Math.round(10 + t * -10);
        const g = Math.round(14 + t * 241);
        const b = Math.round(23 + t * 113);
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(i * cell, p * cell, cell, cell);
      }
    }
    // 当前位置高亮行
    ctx.strokeStyle = "#f59e0b";
    ctx.lineWidth = 2;
    ctx.strokeRect(0, curPos * cell, dim * cell, cell);
  }, [maxPos, dim, curPos]);
  return <canvas ref={ref} width={dim * cell} height={maxPos * cell} style={{ maxWidth: "100%", height: "auto" }} />;
}

export default function PosEncodingViz({ state }: { state: PosEncState }) {
  const { dim, maxPos } = state;
  const [variant, setVariant] = useState<PosEncVariant>("rope");
  const [pos, setPos] = useState(8);

  // RoPE：取第一个二维对的单位向量，按位置旋转
  const ropeVec = useMemo(() => {
    const base = new Array(dim).fill(0);
    base[0] = 1; // 第一个二维对的 x
    const rotated = ropeApply(base, pos);
    return { x: rotated[0], y: rotated[1] };
  }, [dim, pos]);

  // YaRN vs RoPE 角度-位置曲线（取第一个二维对）
  const angleData = useMemo(
    () =>
      Array.from({ length: maxPos }, (_, p) => ({
        pos: p,
        rope: ropeAngle(0, p, dim),
        yarn: yarnAngle(0, p, dim),
      })),
    [dim, maxPos],
  );

  return (
    <div className="flex flex-col items-center gap-4 w-full">
      <div className="flex flex-wrap items-center gap-2">
        {VARIANTS.map((v) => (
          <button key={v.key} onClick={() => setVariant(v.key)}
            className={`px-3 py-1 rounded text-xs font-mono border transition-all ${
              variant === v.key ? "bg-sky-500/25 border-sky-400 text-sky-200"
                : "bg-slate-800/40 border-slate-600 text-slate-400 hover:border-slate-400"}`}>
            {v.label}
          </button>
        ))}
      </div>

      {(variant === "sinusoidal" || variant === "rope") && (
        <label className="flex items-center gap-3 text-xs text-slate-400">
          位置 pos = {pos}
          <input type="range" min={0} max={maxPos - 1} step={1} value={pos}
            onChange={(e) => setPos(Number(e.target.value))} className="w-56" />
        </label>
      )}

      {variant === "sinusoidal" && (
        <div className="flex flex-col items-center gap-2">
          <div className="text-[11px] text-slate-500 font-mono">位置(纵) × 维度(横) 的 sin/cos 指纹 · 橙框=当前 pos</div>
          <Heatmap maxPos={maxPos} dim={dim} curPos={pos} />
        </div>
      )}

      {variant === "rope" && (
        <div className="flex flex-col items-center gap-2">
          <svg width={220} height={220} className="max-w-full">
            <line x1={110} y1={10} x2={110} y2={210} stroke="#334155" />
            <line x1={10} y1={110} x2={210} y2={110} stroke="#334155" />
            <circle cx={110} cy={110} r={90} fill="none" stroke="#1e293b" />
            <line x1={110} y1={110} x2={110 + ropeVec.x * 90} y2={110 - ropeVec.y * 90}
              stroke="#38bdf8" strokeWidth={3} />
            <circle cx={110 + ropeVec.x * 90} cy={110 - ropeVec.y * 90} r={4} fill="#38bdf8" />
          </svg>
          <div className="text-[12px] text-slate-400 max-w-md text-center">
            第一个二维对的查询向量随位置 <span className="font-mono text-sky-300">pos={pos}</span> 旋转。
            关键：注意力分数 <span className="font-mono">q·k</span> 只依赖 <b>相对距离</b>，与绝对位置无关。
          </div>
        </div>
      )}

      {variant === "yarn" && (
        <div className="w-full" style={{ height: 260 }}>
          <ResponsiveContainer>
            <LineChart data={angleData} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="pos" stroke="#64748b" tick={{ fontSize: 11 }} />
              <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
              <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 12 }} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Line type="monotone" dataKey="rope" name="RoPE 角度" stroke="#38bdf8" dot={false} strokeWidth={2} />
              <Line type="monotone" dataKey="yarn" name="YaRN 角度（更慢）" stroke="#f59e0b" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
          <div className="text-[12px] text-slate-400 text-center mt-1">
            YaRN 把旋转频率调慢——同样的位置走过更小的角度，于是短上下文训练的模型能外推到更长上下文。
          </div>
        </div>
      )}

      {variant === "nope" && (
        <div className="text-[13px] text-slate-400 leading-relaxed max-w-xl text-center py-8">
          <div className="text-base text-slate-200 mb-2 font-mono">NoPE：不加任何位置编码</div>
          只靠<b>因果掩码</b>（每个 token 只能看到自己左边）就能隐式获得顺序信息。
          研究发现这种纯解码器在一定规模下也能学到位置概念，挑战了「必须显式编码位置」的假设。
        </div>
      )}
    </div>
  );
}
