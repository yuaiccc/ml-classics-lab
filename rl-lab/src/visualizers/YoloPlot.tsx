// yolo 家族：原图 + 检测框（按置信度阈值显隐）。
import { YoloState } from "@/player/types";

const PALETTE = ["#00ff88", "#00e5ff", "#ffab40", "#b388ff", "#ff5252", "#64ffda", "#ff80ab", "#7c4dff"];
const colorFor = (label: string) => {
  let h = 0;
  for (let i = 0; i < label.length; i++) h = (h * 31 + label.charCodeAt(i)) >>> 0;
  return PALETTE[h % PALETTE.length];
};

export default function YoloPlot({ state }: { state: YoloState }) {
  const { image, imgW, imgH, threshold, boxes } = state;
  const shown = boxes.filter((b) => b.conf >= threshold);

  return (
    <div className="rounded-xl bg-[rgba(10,14,23,0.6)] p-3 flex flex-col items-center">
      <svg
        viewBox={`0 0 ${imgW} ${imgH}`}
        preserveAspectRatio="xMidYMid meet"
        style={{ height: "460px", maxWidth: "100%" }}
        className="rounded-lg"
      >
        <image href={image} x={0} y={0} width={imgW} height={imgH} />
        {shown.map((b, i) => {
          const col = colorFor(b.label);
          const x = b.x * imgW;
          const y = b.y * imgH;
          const w = b.w * imgW;
          const h = b.h * imgH;
          const tag = `${b.label} ${b.conf.toFixed(2)}`;
          const tagW = tag.length * 13 + 14;
          return (
            <g key={i}>
              <rect x={x} y={y} width={w} height={h} fill="none" stroke={col} strokeWidth={4} />
              <rect x={x} y={Math.max(0, y - 30)} width={tagW} height={28} fill={col} rx={3} />
              <text x={x + 6} y={Math.max(0, y - 30) + 20} fill="#0a0e17" fontSize={20} fontWeight="bold" className="font-mono">
                {tag}
              </text>
            </g>
          );
        })}
      </svg>
      <div className="text-xs text-slate-500 mt-2 font-mono">
        置信度阈值 ≥ <span className="text-[#00e5ff]">{threshold.toFixed(2)}</span> · 显示{" "}
        <span className="text-[#00ff88]">{shown.length}</span>/{boxes.length} 个检测框
      </div>
    </div>
  );
}
