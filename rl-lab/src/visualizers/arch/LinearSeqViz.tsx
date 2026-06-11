import { useMemo, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from "recharts";
import { LinearSeqState, LinearSeqVariant } from "./types";
import { attentionMemory, recurrentMemory, scanNorms } from "@/algorithms/arch/linear-seq-fns";

const VARIANTS: { key: LinearSeqVariant; label: string; note: string; approx?: boolean }[] = [
  { key: "lightning", label: "Lightning", note: "线性注意力：用核函数把 softmax 注意力线性化，定长状态、O(L) 计算。" },
  { key: "gateddeltanet", label: "Gated DeltaNet", note: "在线性递归上加门控 + delta 更新规则，提升长程记忆精度。" },
  { key: "kda", label: "KDA", note: "核化 delta 注意力：用核技巧改进 delta 规则的记忆能力。", approx: true },
  { key: "mamba", label: "Mamba", note: "选择性 SSM：让状态转移依赖输入内容，选择性记忆/遗忘。" },
  { key: "mamba2", label: "Mamba-2", note: "把选择性 SSM 与注意力统一（SSD），更高效、更易扩展。" },
  { key: "mamba3", label: "Mamba-3", note: "Mamba 系最新一代的进一步改进。", approx: true },
];

export default function LinearSeqViz({ state }: { state: LinearSeqState }) {
  const [variant, setVariant] = useState<LinearSeqVariant>("mamba");
  const [maxL, setMaxL] = useState(256);
  const stateSize = state.stateSize;
  const cur = VARIANTS.find((v) => v.key === variant)!;

  const memData = useMemo(
    () =>
      Array.from({ length: 17 }, (_, i) => {
        const L = Math.round((maxL * i) / 16);
        return { L, attention: attentionMemory(L), linear: recurrentMemory(stateSize) };
      }),
    [maxL, stateSize],
  );

  const norms = useMemo(() => scanNorms(state.tokens, stateSize), [state.tokens, stateSize]);
  const scanData = norms.map((n, i) => ({ step: i, norm: Number(n.toFixed(3)) }));

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

      <div className="text-[12px] text-slate-400 max-w-xl text-center">
        {cur.note}{cur.approx && <span className="text-amber-400/80"> · 按公开描述近似演示</span>}
      </div>

      <label className="flex items-center gap-3 text-xs text-slate-400">
        最大序列长度 {maxL}
        <input type="range" min={64} max={1024} step={64} value={maxL}
          onChange={(e) => setMaxL(Number(e.target.value))} className="w-56" />
      </label>

      <div className="w-full" style={{ height: 220 }}>
        <ResponsiveContainer>
          <LineChart data={memData} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="L" stroke="#64748b" tick={{ fontSize: 11 }} label={{ value: "序列长度 L", position: "insideBottom", offset: -2, fontSize: 11, fill: "#64748b" }} />
            <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
            <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 12 }} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Line type="monotone" dataKey="attention" name="注意力 KV-cache（∝L）" stroke="#ff5252" dot={false} strokeWidth={2} />
            <Line type="monotone" dataKey="linear" name="线性/SSM 定长状态" stroke="#38bdf8" dot={false} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="w-full" style={{ height: 150 }}>
        <div className="text-[11px] text-slate-500 mb-1">扫描：固定大小状态的范数随 token 流更新（{variant}）</div>
        <ResponsiveContainer>
          <LineChart data={scanData} margin={{ top: 4, right: 16, bottom: 4, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="step" stroke="#64748b" tick={{ fontSize: 11 }} />
            <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
            <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 12 }} />
            <Line type="monotone" dataKey="norm" name="状态范数" stroke="#a855f7" dot={false} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
