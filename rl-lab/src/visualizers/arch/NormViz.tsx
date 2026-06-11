import { useMemo, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from "recharts";
import { NormState, NormVariant } from "./types";
import { layerNorm, rmsNorm, mean, rms, cosine } from "@/algorithms/arch/normalization-fns";

const VARIANTS: { key: NormVariant; label: string }[] = [
  { key: "layernorm", label: "LayerNorm" },
  { key: "rmsnorm", label: "RMSNorm" },
  { key: "qknorm", label: "QKNorm" },
  { key: "prepost", label: "Pre/Post-Norm" },
];

function VecRow({ v, normed }: { v: number[]; normed: number[] }) {
  return (
    <tr className="font-mono text-[11px]">
      <td className="px-2 py-1 text-slate-500">[{v.map((x) => x.toFixed(1)).join(", ")}]</td>
      <td className="px-2 py-1 text-sky-300">[{normed.map((x) => x.toFixed(2)).join(", ")}]</td>
      <td className="px-2 py-1 text-slate-400">{mean(normed).toFixed(2)}</td>
      <td className="px-2 py-1 text-slate-400">{rms(normed).toFixed(2)}</td>
      <td className="px-2 py-1 text-amber-400">{cosine(normed, v).toFixed(3)}</td>
    </tr>
  );
}

export default function NormViz({ state }: { state: NormState }) {
  const [variant, setVariant] = useState<NormVariant>("layernorm");
  const vectors = state.vectors;

  const normFn = variant === "rmsnorm" ? rmsNorm : layerNorm;

  const depthData = useMemo(
    () =>
      Array.from({ length: 24 }, (_, d) => ({
        depth: d,
        post: Math.pow(1.18, d),
        pre: 1 + Math.log1p(d) * 0.6,
      })),
    [],
  );

  const qkData = useMemo(
    () =>
      Array.from({ length: 12 }, (_, i) => {
        const scale = 1 + i * 0.6;
        const d = 8;
        const raw = scale * scale * d * 0.5;
        const normed = 1 * 1 * d * 0.5;
        return { scale: Number(scale.toFixed(1)), raw, normed };
      }),
    [],
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

      {(variant === "layernorm" || variant === "rmsnorm") && (
        <div className="w-full max-w-xl overflow-x-auto">
          <table className="w-full text-left">
            <thead>
              <tr className="text-[11px] text-slate-500">
                <th className="px-2 py-1">原向量</th>
                <th className="px-2 py-1">归一化后</th>
                <th className="px-2 py-1">均值</th>
                <th className="px-2 py-1">RMS</th>
                <th className="px-2 py-1">余弦(原)</th>
              </tr>
            </thead>
            <tbody>
              {vectors.map((v, i) => (
                <VecRow key={i} v={v} normed={normFn(v)} />
              ))}
            </tbody>
          </table>
          <div className="text-[12px] text-slate-400 mt-2 px-2">
            {variant === "layernorm"
              ? "LayerNorm：归一化后每个向量均值≈0；因为先去了均值，方向(余弦)通常≠1。"
              : "RMSNorm：只按 RMS 缩放，方向不变(余弦≈1.000)，也不强行把均值压到 0——更省、效果不输。"}
          </div>
        </div>
      )}

      {variant === "qknorm" && (
        <div className="w-full" style={{ height: 260 }}>
          <ResponsiveContainer>
            <LineChart data={qkData} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="scale" stroke="#64748b" tick={{ fontSize: 11 }} label={{ value: "Q/K 尺度", position: "insideBottom", offset: -2, fontSize: 11, fill: "#64748b" }} />
              <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
              <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 12 }} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Line type="monotone" dataKey="raw" name="未归一化 logit" stroke="#ff5252" dot={false} strokeWidth={2} />
              <Line type="monotone" dataKey="normed" name="QKNorm 后 logit" stroke="#38bdf8" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
          <div className="text-[12px] text-slate-400 text-center mt-1">
            不归一化时注意力 logit 随 Q/K 尺度平方膨胀→softmax 饱和、梯度消失；QKNorm 把 Q/K 拉回单位尺度，logit 受控。
          </div>
        </div>
      )}

      {variant === "prepost" && (
        <div className="w-full" style={{ height: 260 }}>
          <ResponsiveContainer>
            <LineChart data={depthData} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="depth" stroke="#64748b" tick={{ fontSize: 11 }} label={{ value: "层深", position: "insideBottom", offset: -2, fontSize: 11, fill: "#64748b" }} />
              <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
              <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 12 }} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Line type="monotone" dataKey="post" name="Post-Norm 残差流幅度" stroke="#ff5252" dot={false} strokeWidth={2} />
              <Line type="monotone" dataKey="pre" name="Pre-Norm 残差流幅度" stroke="#38bdf8" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
          <div className="text-[12px] text-slate-400 text-center mt-1">
            Pre-Norm（归一化放残差之前）让残差流幅度受控、深层好训，是现代 Transformer 默认；Post-Norm 幅度易滚雪球、难训但表达力强。
          </div>
        </div>
      )}
    </div>
  );
}
