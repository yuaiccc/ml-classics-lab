import { useMemo, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from "recharts";
import { ResidualState, ResidualVariant } from "./types";
import { gradMagnitude } from "@/algorithms/arch/residual-fns";

const VARIANTS: { key: ResidualVariant; label: string }[] = [
  { key: "plain", label: "Plain（无捷径）" },
  { key: "rc", label: "RC (ResNet)" },
  { key: "hc", label: "HC" },
  { key: "mhc", label: "mHC" },
  { key: "attnres", label: "AttnResidual" },
];

export default function ResidualViz({ state }: { state: ResidualState }) {
  const [variant, setVariant] = useState<ResidualVariant>("rc");
  const [nLayers, setNLayers] = useState(state.nLayers);
  const [streams, setStreams] = useState(4);

  const data = useMemo(
    () =>
      Array.from({ length: nLayers + 1 }, (_, d) => ({
        depth: d,
        cur: gradMagnitude(variant, d, nLayers, 0.8, streams),
        plain: gradMagnitude("plain", d, nLayers, 0.8),
      })),
    [variant, nLayers, streams],
  );

  const multiStream = variant === "hc" || variant === "mhc";

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

      <div className="flex flex-wrap gap-5 text-xs text-slate-400">
        <label className="flex items-center gap-2">层数 {nLayers}
          <input type="range" min={8} max={48} step={2} value={nLayers}
            onChange={(e) => setNLayers(Number(e.target.value))} className="w-32" />
        </label>
        {multiStream && (
          <label className="flex items-center gap-2">残差流数 {streams}
            <input type="range" min={2} max={8} step={1} value={streams}
              onChange={(e) => setStreams(Number(e.target.value))} className="w-28" />
          </label>
        )}
      </div>

      <div className="w-full" style={{ height: 240 }}>
        <ResponsiveContainer>
          <LineChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="depth" stroke="#64748b" tick={{ fontSize: 11 }} label={{ value: "层深（0=输入）", position: "insideBottom", offset: -2, fontSize: 11, fill: "#64748b" }} />
            <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
            <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 12 }} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Line type="monotone" dataKey="cur" name="当前变体梯度幅度" stroke="#38bdf8" dot={false} strokeWidth={2} />
            <Line type="monotone" dataKey="plain" name="Plain 参照（消失）" stroke="#ff5252" dot={false} strokeWidth={1.5} strokeDasharray="4 3" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <svg width={320} height={120} className="max-w-full">
        <rect x={130} y={20} width={60} height={28} rx={4} fill="#1e293b" stroke="#475569" />
        <text x={160} y={38} textAnchor="middle" fontSize={11} fill="#94a3b8">f(x)</text>
        <line x1={40} y1={34} x2={130} y2={34} stroke="#64748b" strokeWidth={2} markerEnd="url(#a)" />
        <line x1={190} y1={34} x2={280} y2={34} stroke="#64748b" strokeWidth={2} />
        {variant !== "plain" && (
          <>
            <path d={`M70,34 Q70,90 160,90 Q250,90 250,42`} fill="none" stroke="#38bdf8" strokeWidth={2} />
            <text x={160} y={108} textAnchor="middle" fontSize={10} fill="#38bdf8">
              {multiStream ? `${streams} 条并行残差捷径 (+)` : "恒等捷径 skip (+)"}
            </text>
          </>
        )}
        <circle cx={250} cy={34} r={10} fill="#0f172a" stroke="#38bdf8" />
        <text x={250} y={38} textAnchor="middle" fontSize={12} fill="#38bdf8">+</text>
        <defs>
          <marker id="a" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 Z" fill="#64748b" />
          </marker>
        </defs>
      </svg>
      {variant === "plain" && (
        <div className="text-[12px] text-amber-400/80 text-center">Plain 没有捷径——梯度只能走 f(x) 这条路，按层数指数衰减。</div>
      )}
    </div>
  );
}
