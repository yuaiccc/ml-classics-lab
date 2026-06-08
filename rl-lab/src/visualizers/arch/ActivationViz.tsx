import { useMemo, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Legend,
} from "recharts";
import { ActivationState, ActivationVariant } from "./types";
import { relu, dRelu, silu, dSilu, swish, dSwish, gelu, dGelu } from "@/algorithms/arch/activation-fns";

const VARIANTS: { key: ActivationVariant; label: string }[] = [
  { key: "relu", label: "ReLU" },
  { key: "gelu", label: "GELU" },
  { key: "silu", label: "SiLU" },
  { key: "swish", label: "Swish(β)" },
  { key: "swiglu", label: "SwiGLU" },
];

function val(v: ActivationVariant, x: number, beta: number): number {
  switch (v) {
    case "relu": return relu(x);
    case "gelu": return gelu(x);
    case "silu": return silu(x);
    case "swish": return swish(x, beta);
    case "swiglu": return silu(x); // 门控的「激活支」用 SiLU；门控演示见下方面板
  }
}
function deriv(v: ActivationVariant, x: number, beta: number): number {
  switch (v) {
    case "relu": return dRelu(x);
    case "gelu": return dGelu(x);
    case "silu": return dSilu(x);
    case "swish": return dSwish(x, beta);
    case "swiglu": return dSilu(x);
  }
}

export default function ActivationViz({ state }: { state: ActivationState }) {
  const [variant, setVariant] = useState<ActivationVariant>("silu");
  const [beta, setBeta] = useState(1);
  const xs = state.xs;

  const data = useMemo(
    () => xs.map((x) => ({ x: Number(x.toFixed(3)), y: val(variant, x, beta), dy: deriv(variant, x, beta) })),
    [xs, variant, beta],
  );

  return (
    <div className="flex flex-col items-center gap-4 w-full">
      <div className="flex flex-wrap items-center gap-2">
        {VARIANTS.map((v) => (
          <button
            key={v.key}
            onClick={() => setVariant(v.key)}
            className={`px-3 py-1 rounded text-xs font-mono border transition-all ${
              variant === v.key
                ? "bg-sky-500/25 border-sky-400 text-sky-200"
                : "bg-slate-800/40 border-slate-600 text-slate-400 hover:border-slate-400"
            }`}
          >
            {v.label}
          </button>
        ))}
      </div>

      {(variant === "swish") && (
        <label className="flex items-center gap-3 text-xs text-slate-400">
          β = {beta.toFixed(2)}
          <input type="range" min={0.1} max={5} step={0.1} value={beta}
            onChange={(e) => setBeta(Number(e.target.value))} className="w-48" />
        </label>
      )}

      <div className="w-full" style={{ height: 280 }}>
        <ResponsiveContainer>
          <LineChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="x" type="number" domain={[-6, 6]} stroke="#64748b" tick={{ fontSize: 11 }} />
            <YAxis stroke="#64748b" tick={{ fontSize: 11 }} domain={[-2, 6]} />
            <ReferenceLine x={0} stroke="#475569" />
            <ReferenceLine y={0} stroke="#475569" />
            <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 12 }} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Line type="monotone" dataKey="y" name="激活 f(x)" stroke="#38bdf8" dot={false} strokeWidth={2} />
            <Line type="monotone" dataKey="dy" name="导数 f'(x)" stroke="#f59e0b" dot={false} strokeWidth={1.5} strokeDasharray="4 3" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {variant === "swiglu" && (
        <div className="text-[12px] text-slate-400 leading-relaxed max-w-xl text-center">
          <span className="font-mono text-sky-300">SwiGLU(x) = SiLU(x·W) ⊙ (x·V)</span>
          ：把输入投影成两路，一路过 SiLU 当「门」，逐元素乘到另一「值」路上——
          门接近 0 就压制、接近线性就放行。现代 LLM（LLaMA 系）FFN 的标配。
        </div>
      )}
    </div>
  );
}
