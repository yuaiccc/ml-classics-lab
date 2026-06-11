import { useEffect, useMemo, useRef, useState } from "react";
import { SparseAttnState, SparseAttnVariant } from "./types";
import { attends, attendedFraction } from "@/algorithms/arch/sparse-attention-fns";
import AttentionHeatmap from "@/visualizers/AttentionHeatmap";
import imdbTransformer from "@/data/frames/imdb-transformer.json";
import type { AttentionState } from "../lab2-types";

const VARIANTS: { key: SparseAttnVariant; label: string; approx?: boolean }[] = [
  { key: "full", label: "Full（因果全连接）" },
  { key: "swa", label: "SWA 滑动窗口" },
  { key: "sparse", label: "Sparse Transformer" },
  { key: "dsa", label: "DSA", approx: true },
  { key: "csa", label: "CSA/HCA", approx: true },
];

// 仓库已有的本地真训 IMDb Transformer 的真实注意力（末帧）。
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const REAL_ATTN = (imdbTransformer as any).frames[(imdbTransformer as any).frames.length - 1].state.data as AttentionState;

function MaskCanvas({ variant, L, window }: { variant: SparseAttnVariant; L: number; window: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const cell = Math.max(4, Math.floor(360 / L));
  useEffect(() => {
    const ctx = ref.current?.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, L * cell, L * cell);
    for (let i = 0; i < L; i++) {
      for (let j = 0; j < L; j++) {
        const on = attends(variant, i, j, L, window);
        ctx.fillStyle = on ? "#38bdf8" : "#1e293b";
        ctx.fillRect(j * cell, i * cell, cell - 1, cell - 1);
      }
    }
  }, [variant, L, window, cell]);
  return <canvas ref={ref} width={L * cell} height={L * cell} style={{ maxWidth: "100%", height: "auto" }} />;
}

export default function SparseAttnViz({ state }: { state: SparseAttnState }) {
  const [variant, setVariant] = useState<SparseAttnVariant>("swa");
  const [L, setL] = useState(state.seqLen);
  const [window, setWindow] = useState(4);
  const [showReal, setShowReal] = useState(false);
  const cur = VARIANTS.find((v) => v.key === variant)!;
  const frac = useMemo(() => attendedFraction(variant, L, window), [variant, L, window]);

  return (
    <div className="flex flex-col items-center gap-4 w-full">
      <div className="flex flex-wrap items-center gap-2">
        {VARIANTS.map((v) => (
          <button key={v.key} onClick={() => { setVariant(v.key); setShowReal(false); }}
            className={`px-3 py-1 rounded text-xs font-mono border transition-all ${
              !showReal && variant === v.key ? "bg-sky-500/25 border-sky-400 text-sky-200"
                : "bg-slate-800/40 border-slate-600 text-slate-400 hover:border-slate-400"}`}>
            {v.label}
          </button>
        ))}
        <button onClick={() => setShowReal((s) => !s)}
          className={`px-3 py-1 rounded text-xs font-mono border transition-all ${
            showReal ? "bg-purple-500/25 border-purple-400 text-purple-200"
              : "bg-slate-800/40 border-slate-600 text-slate-400 hover:border-slate-400"}`}>
          真实样本
        </button>
      </div>

      {showReal ? (
        <div className="flex flex-col items-center gap-2">
          <div className="text-[12px] text-purple-300/90 max-w-xl text-center">
            本地真训 IMDb Transformer 的真实注意力（非掩码图案）——可见注意力天然集中在少数 token 上，这正是稀疏注意力「只算重要格子也够用」的依据。
          </div>
          <AttentionHeatmap state={REAL_ATTN} />
        </div>
      ) : (
        <>
          <div className="text-[12px] text-slate-400 max-w-xl text-center">
            掩码图案（亮=被计算）。{cur.approx && <span className="text-amber-400/80">该变体按公开描述近似演示。</span>}
          </div>
          <div className="flex flex-wrap gap-5 text-xs text-slate-400">
            <label className="flex items-center gap-2">序列长 {L}
              <input type="range" min={8} max={48} step={2} value={L}
                onChange={(e) => setL(Number(e.target.value))} className="w-32" />
            </label>
            {(variant === "swa" || variant === "sparse" || variant === "dsa" || variant === "csa") && (
              <label className="flex items-center gap-2">窗口 {window}
                <input type="range" min={1} max={12} step={1} value={window}
                  onChange={(e) => setWindow(Number(e.target.value))} className="w-28" />
              </label>
            )}
          </div>
          <MaskCanvas variant={variant} L={L} window={window} />
          <div className="text-xs text-slate-400">
            被计算格子（FLOPs 占比）：<span className="font-mono text-sky-300">{(frac * 100).toFixed(1)}%</span>
            <span className="text-slate-600"> · 行=query，列=key，因果下三角</span>
          </div>
        </>
      )}
    </div>
  );
}
