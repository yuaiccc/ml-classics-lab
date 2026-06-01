// cnn 家族：输入图 + 第一层卷积核 + 该样本的激活图，三栏热力图。
// 卷积核随训练 epoch 从噪声变成边缘检测器。
import { CnnState } from "@/player/types";

// 单张热力图（SVG 方格）。diverging=true 时用双色（正橙/负青）显示可正可负的卷积核。
function Heatmap({
  m,
  cell,
  diverging = false,
}: {
  m: number[][];
  cell: number;
  diverging?: boolean;
}) {
  const rows = m.length;
  const cols = m[0].length;
  let min = Infinity;
  let max = -Infinity;
  let maxAbs = 1e-9;
  for (const row of m)
    for (const v of row) {
      if (v < min) min = v;
      if (v > max) max = v;
      if (Math.abs(v) > maxAbs) maxAbs = Math.abs(v);
    }
  const span = max - min || 1;

  const color = (v: number) => {
    if (diverging) {
      const t = Math.min(1, Math.abs(v) / maxAbs);
      return v >= 0 ? `rgba(255,171,64,${t})` : `rgba(0,229,255,${t})`;
    }
    const t = (v - min) / span;
    return `rgba(0,255,136,${0.06 + 0.94 * t})`;
  };

  return (
    <svg
      width={cols * cell}
      height={rows * cell}
      className="rounded-sm bg-[#0a0e17] shrink-0"
      shapeRendering="crispEdges"
    >
      {m.map((row, r) =>
        row.map((v, c) => (
          <rect key={`${r}-${c}`} x={c * cell} y={r * cell} width={cell} height={cell} fill={color(v)} />
        ))
      )}
    </svg>
  );
}

export default function CnnPlot({ state }: { state: CnnState }) {
  const { input, filters, activations, label, pred } = state;
  const names = ["横线", "竖线"];
  return (
    <div className="rounded-xl bg-[rgba(10,14,23,0.6)] p-5 flex flex-col gap-5">
      <div className="flex flex-wrap items-start gap-6">
        {/* 输入 */}
        <div className="flex flex-col gap-1.5">
          <div className="text-xs text-slate-500">输入样本</div>
          <Heatmap m={input} cell={12} />
          <div className="text-xs font-mono">
            <span className="text-slate-500">真实 </span>
            <span className="text-[#00e5ff]">{names[label] ?? label}</span>
            <span className="text-slate-600"> · </span>
            <span className="text-slate-500">预测 </span>
            <span className={pred === label ? "text-[#00ff88]" : "text-[#ff5252]"}>
              {names[pred] ?? pred}
            </span>
          </div>
        </div>

        {/* 卷积核 */}
        <div className="flex flex-col gap-1.5">
          <div className="text-xs text-slate-500">第一层卷积核 ×{filters.length}（橙=正 / 青=负）</div>
          <div className="flex gap-2">
            {filters.map((f, i) => (
              <Heatmap key={i} m={f} cell={14} diverging />
            ))}
          </div>
        </div>
      </div>

      {/* 激活图 */}
      <div className="flex flex-col gap-1.5">
        <div className="text-xs text-slate-500">激活图（每个卷积核扫过输入后的响应，越亮=越匹配）</div>
        <div className="flex flex-wrap gap-2">
          {activations.map((a, i) => (
            <Heatmap key={i} m={a} cell={9} />
          ))}
        </div>
      </div>
    </div>
  );
}
