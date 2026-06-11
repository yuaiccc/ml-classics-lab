import { useState } from "react";
import { AttnKVState, KVVariant } from "./types";
import { kvHeads, cacheTotal } from "@/algorithms/arch/kv-cache";

const VARIANTS: { key: KVVariant; label: string }[] = [
  { key: "mha", label: "MHA" },
  { key: "mqa", label: "MQA" },
  { key: "gqa", label: "GQA" },
  { key: "mla", label: "MLA" },
];

export default function AttnKVViz({ state }: { state: AttnKVState }) {
  const [variant, setVariant] = useState<KVVariant>("gqa");
  const [nHeads, setNHeads] = useState(state.config.nHeads);
  const [nGroups, setNGroups] = useState(state.config.nGroups);
  const [seqLen, setSeqLen] = useState(state.config.seqLen);
  const cfg = { ...state.config, nHeads, nGroups, seqLen };

  // 每个 Q 头映射到哪个 KV 头（索引）
  const kvOf = (qi: number): number => {
    switch (variant) {
      case "mha": return qi;
      case "mqa": return 0;
      case "gqa": return Math.floor(qi / Math.ceil(nHeads / Math.max(1, nGroups)));
      case "mla": return 0; // 潜向量，单独画
    }
  };
  const nKV = kvHeads(variant, cfg);

  // KV-cache 对比（相对 MHA 的百分比）
  const totals = VARIANTS.map((v) => ({ key: v.key, label: v.label, total: cacheTotal(v.key, cfg) }));
  const maxTotal = Math.max(...totals.map((t) => t.total));

  const W = 460, qY = 40, kvY = 200, headW = 34, gap = 12;
  const qX = (i: number) => 30 + i * (headW + gap);
  const kvX = (i: number) => 30 + i * (headW + gap) + (variant === "mqa" || variant === "mla" ? (W - 60) / 2 - headW / 2 : 0);

  return (
    <div className="flex flex-col items-center gap-5 w-full">
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
        <label className="flex items-center gap-2">Q 头数 {nHeads}
          <input type="range" min={2} max={12} step={1} value={nHeads}
            onChange={(e) => setNHeads(Number(e.target.value))} className="w-32" />
        </label>
        {variant === "gqa" && (
          <label className="flex items-center gap-2">分组数 {nGroups}
            <input type="range" min={1} max={nHeads} step={1} value={Math.min(nGroups, nHeads)}
              onChange={(e) => setNGroups(Number(e.target.value))} className="w-32" />
          </label>
        )}
        <label className="flex items-center gap-2">序列长 {seqLen}
          <input type="range" min={256} max={8192} step={256} value={seqLen}
            onChange={(e) => setSeqLen(Number(e.target.value))} className="w-32" />
        </label>
      </div>

      {/* Q→KV 头映射示意 */}
      <svg width={W} height={260} className="max-w-full">
        {variant !== "mla" &&
          Array.from({ length: nHeads }).map((_, qi) => {
            const kx = kvX(kvOf(qi)) + headW / 2;
            return <line key={qi} x1={qX(qi) + headW / 2} y1={qY + 26} x2={kx} y2={kvY} stroke="#475569" strokeWidth={1} />;
          })}
        {variant === "mla" &&
          Array.from({ length: nHeads }).map((_, qi) => (
            <line key={qi} x1={qX(qi) + headW / 2} y1={qY + 26} x2={W / 2} y2={kvY} stroke="#a855f7" strokeWidth={1} />
          ))}
        {Array.from({ length: nHeads }).map((_, qi) => (
          <g key={qi}>
            <rect x={qX(qi)} y={qY} width={headW} height={26} rx={4} fill="#0ea5e9" opacity={0.8} />
            <text x={qX(qi) + headW / 2} y={qY + 17} textAnchor="middle" fontSize={10} fill="#fff">Q{qi}</text>
          </g>
        ))}
        {variant !== "mla" &&
          Array.from({ length: nKV }).map((_, ki) => (
            <g key={ki}>
              <rect x={kvX(ki)} y={kvY} width={headW} height={26} rx={4} fill="#f59e0b" opacity={0.85} />
              <text x={kvX(ki) + headW / 2} y={kvY + 17} textAnchor="middle" fontSize={10} fill="#fff">KV{ki}</text>
            </g>
          ))}
        {variant === "mla" && (
          <g>
            <rect x={W / 2 - headW} y={kvY} width={headW * 2} height={26} rx={4} fill="#a855f7" opacity={0.85} />
            <text x={W / 2} y={kvY + 17} textAnchor="middle" fontSize={10} fill="#fff">潜向量 latent</text>
          </g>
        )}
        <text x={30} y={qY - 8} fontSize={11} fill="#94a3b8">查询头 Query heads</text>
        <text x={30} y={kvY - 8} fontSize={11} fill="#94a3b8">KV 头（缓存）</text>
      </svg>

      {/* KV-cache 显存对比条 */}
      <div className="w-full max-w-md flex flex-col gap-2">
        <div className="text-xs text-slate-400">KV-cache 显存（相对最大值）· 序列长 {seqLen}</div>
        {totals.map((t) => (
          <div key={t.key} className="flex items-center gap-2">
            <span className={`w-12 text-[11px] font-mono ${t.key === variant ? "text-sky-300" : "text-slate-500"}`}>{t.label}</span>
            <div className="flex-1 h-4 bg-slate-800 rounded overflow-hidden">
              <div className="h-full rounded" style={{ width: `${(t.total / maxTotal) * 100}%`, background: t.key === variant ? "#38bdf8" : "#475569" }} />
            </div>
            <span className="w-16 text-right text-[11px] font-mono text-slate-400">{(t.total / 1e6).toFixed(2)}M</span>
          </div>
        ))}
      </div>
    </div>
  );
}
