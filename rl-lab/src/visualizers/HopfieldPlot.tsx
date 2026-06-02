// hopfield 家族：当前状态网格（大）+ 目标图案（小，参考）。
import { HopfieldState } from "@/player/types";

function Grid({ cells, size, cell }: { cells: number[]; size: number; cell: number }) {
  return (
    <svg width={size * cell} height={size * cell} className="rounded-md bg-[#0a0e17]" shapeRendering="crispEdges">
      {cells.map((v, i) => {
        const r = Math.floor(i / size);
        const c = i % size;
        return (
          <rect
            key={i}
            x={c * cell}
            y={r * cell}
            width={cell - 1}
            height={cell - 1}
            fill={v > 0 ? "#00ff88" : "rgba(148,163,184,0.12)"}
          />
        );
      })}
    </svg>
  );
}

export default function HopfieldPlot({ state }: { state: HopfieldState }) {
  const { cells, target, size } = state;
  return (
    <div className="rounded-xl bg-[rgba(10,14,23,0.6)] p-5 flex items-end gap-8 justify-center">
      <div className="flex flex-col items-center gap-2">
        <div className="text-xs text-slate-500">当前状态（回忆中）</div>
        <Grid cells={cells} size={size} cell={30} />
      </div>
      <div className="flex flex-col items-center gap-2">
        <div className="text-xs text-slate-500">目标图案</div>
        <Grid cells={target} size={size} cell={14} />
      </div>
    </div>
  );
}
