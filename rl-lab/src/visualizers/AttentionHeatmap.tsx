import { useRef, useEffect, useState, useMemo } from "react";
import { AttentionState } from "./lab2-types";

interface Props {
  state: AttentionState;
}

const CELL = 26;
const LABEL_PAD = 92;

// 把 [0,1] 注意力权重映射成颜色（深底 → 青绿高亮）
function colorFor(v: number): string {
  const t = Math.max(0, Math.min(1, v));
  const r = Math.round(10 + t * (0 - 10));
  const g = Math.round(14 + t * (255 - 14));
  const b = Math.round(23 + t * (136 - 23));
  return `rgb(${r},${g},${b})`;
}

export default function AttentionHeatmap({ state }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [layer, setLayer] = useState(0);
  const [head, setHead] = useState(0);

  const tokens = state.tokens;
  const nLayers = state.attention.length;
  const nHeads = nLayers > 0 ? state.attention[0].length : 0;
  const L = tokens.length;

  // 防止帧/数据变化后索引越界
  const safeLayer = Math.min(layer, Math.max(0, nLayers - 1));
  const safeHead = Math.min(head, Math.max(0, nHeads - 1));
  const matrix = useMemo(
    () => state.attention[safeLayer]?.[safeHead] ?? [],
    [state.attention, safeLayer, safeHead],
  );

  const gridW = LABEL_PAD + L * CELL;
  const gridH = LABEL_PAD + L * CELL;

  useEffect(() => {
    const ctx = canvasRef.current?.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, gridW, gridH);
    ctx.fillStyle = "rgba(250, 249, 245, 1)";
    ctx.fillRect(0, 0, gridW, gridH);

    // 单元格（行=query token，列=key token）
    for (let i = 0; i < L; i++) {
      for (let j = 0; j < L; j++) {
        const v = matrix[i]?.[j] ?? 0;
        ctx.fillStyle = colorFor(v);
        ctx.fillRect(LABEL_PAD + j * CELL, LABEL_PAD + i * CELL, CELL - 1, CELL - 1);
      }
    }

    // 列标签（key tokens，竖排在顶部）
    ctx.font = "11px 'JetBrains Mono', monospace";
    ctx.fillStyle = "rgba(107, 106, 101, 0.85)";
    ctx.textBaseline = "middle";
    for (let j = 0; j < L; j++) {
      ctx.save();
      ctx.translate(LABEL_PAD + j * CELL + CELL / 2, LABEL_PAD - 6);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "left";
      ctx.fillText(tokens[j] ?? "", 0, 0);
      ctx.restore();
    }
    // 行标签（query tokens，左侧）
    ctx.textAlign = "right";
    for (let i = 0; i < L; i++) {
      ctx.fillText(tokens[i] ?? "", LABEL_PAD - 6, LABEL_PAD + i * CELL + CELL / 2);
    }
  }, [matrix, tokens, L, gridW, gridH]);

  return (
    <div className="flex flex-col items-center gap-3">
      <div className="flex items-center gap-4 text-xs">
        <label className="flex items-center gap-2 text-stone-600">
          层
          <select
            value={safeLayer}
            onChange={(e) => setLayer(Number(e.target.value))}
            className="bg-stone-200 border border-stone-300 rounded px-2 py-1 text-stone-800 font-mono"
          >
            {Array.from({ length: nLayers }, (_, i) => (
              <option key={i} value={i}>
                {i + 1}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2 text-stone-600">
          头
          <select
            value={safeHead}
            onChange={(e) => setHead(Number(e.target.value))}
            className="bg-stone-200 border border-stone-300 rounded px-2 py-1 text-stone-800 font-mono"
          >
            {Array.from({ length: nHeads }, (_, i) => (
              <option key={i} value={i}>
                {i + 1}
              </option>
            ))}
          </select>
        </label>
        <span className="text-stone-500 font-mono">行=query · 列=key · 亮=高注意力</span>
      </div>
      <canvas
        ref={canvasRef}
        width={gridW}
        height={gridH}
        className="rounded-lg border border-stone-200/50"
        style={{ maxWidth: "100%", height: "auto" }}
      />
    </div>
  );
}
