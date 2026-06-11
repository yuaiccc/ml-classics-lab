import { useMemo, useState } from "react";
import { MoEState, MoEVariant } from "./types";
import { routedExperts, activeFraction } from "@/algorithms/arch/moe-fns";

const VARIANTS: { key: MoEVariant; label: string }[] = [
  { key: "dense", label: "Dense" },
  { key: "moe", label: "MoE (top-k)" },
  { key: "switch", label: "Switch (top-1)" },
  { key: "deepseek", label: "DeepSeekMoE" },
];

export default function MoEViz({ state }: { state: MoEState }) {
  const [variant, setVariant] = useState<MoEVariant>("moe");
  const [nExperts, setNExperts] = useState(state.nExperts);
  const [topK, setTopK] = useState(state.topK);
  const [tokenIdx, setTokenIdx] = useState(0);
  const tokens = state.tokens;
  const nShared = state.nShared;

  const curToken = tokens[tokenIdx];
  const active = useMemo(
    () => new Set(routedExperts(variant, curToken, nExperts, topK)),
    [variant, curToken, nExperts, topK],
  );
  const sharedActive = variant === "deepseek";

  const load = useMemo(() => {
    const counts = new Array(nExperts).fill(0);
    tokens.forEach((t) => routedExperts(variant, t, nExperts, topK).forEach((e) => (counts[e] += 1)));
    return counts;
  }, [tokens, variant, nExperts, topK]);
  const maxLoad = Math.max(1, ...load);

  const frac = activeFraction(variant, nExperts, topK, nShared);

  const W = 480, eW = Math.min(44, (W - 40) / nExperts - 6), gap = 6, eY = 150;
  const eX = (i: number) => 20 + i * (eW + gap);

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
        <label className="flex items-center gap-2">专家数 {nExperts}
          <input type="range" min={4} max={12} step={1} value={nExperts}
            onChange={(e) => setNExperts(Number(e.target.value))} className="w-28" />
        </label>
        {(variant === "moe" || variant === "deepseek") && (
          <label className="flex items-center gap-2">top-k {topK}
            <input type="range" min={1} max={Math.min(4, nExperts)} step={1} value={topK}
              onChange={(e) => setTopK(Number(e.target.value))} className="w-24" />
          </label>
        )}
        <label className="flex items-center gap-2">token #{tokenIdx} (值={curToken})
          <input type="range" min={0} max={tokens.length - 1} step={1} value={tokenIdx}
            onChange={(e) => setTokenIdx(Number(e.target.value))} className="w-28" />
        </label>
      </div>

      <svg width={W} height={200} className="max-w-full">
        <g>
          <rect x={W / 2 - 24} y={20} width={48} height={26} rx={4} fill="#0ea5e9" />
          <text x={W / 2} y={37} textAnchor="middle" fontSize={11} fill="#fff">tok {curToken}</text>
        </g>
        {Array.from({ length: nExperts }).map((_, i) => {
          const on = active.has(i);
          return (
            <g key={i}>
              {on && <line x1={W / 2} y1={46} x2={eX(i) + eW / 2} y2={eY} stroke="#38bdf8" strokeWidth={1.5} />}
              <rect x={eX(i)} y={eY} width={eW} height={28} rx={4}
                fill={on ? "#38bdf8" : "#1e293b"} stroke={on ? "#7dd3fc" : "#334155"} />
              <text x={eX(i) + eW / 2} y={eY + 18} textAnchor="middle" fontSize={10} fill={on ? "#fff" : "#64748b"}>E{i}</text>
            </g>
          );
        })}
        {sharedActive && (
          <text x={20} y={eY - 8} fontSize={10} fill="#a855f7">+ {nShared} 个常驻共享专家始终参与</text>
        )}
      </svg>

      <div className="w-full max-w-md flex flex-col gap-1">
        <div className="text-xs text-slate-400">
          每 token 稀疏激活：<span className="font-mono text-sky-300">{(frac * 100).toFixed(0)}%</span> 的专家
        </div>
        <div className="text-xs text-slate-400 mt-2">负载均衡（各专家在整条序列被选次数）</div>
        <div className="flex items-end gap-1 h-16">
          {load.map((c, i) => (
            <div key={i} className="flex-1 bg-slate-700 rounded-t" style={{ height: `${(c / maxLoad) * 100}%`, minHeight: 2 }} title={`E${i}: ${c}`} />
          ))}
        </div>
      </div>
    </div>
  );
}
