// versions 家族：版本演进时间轴 + 当前版本详情卡 + Wikipedia 权威定义。
import { VersionsState } from "@/player/types";
import { BookMarked, Calendar, Users, Lightbulb } from "lucide-react";

const W = 720;

export default function VersionsPlot({ state }: { state: VersionsState }) {
  const { items, current, wikiDef, wikiUrl } = state;
  const cur = items[current];
  const n = items.length;
  const pad = 30;
  const x = (i: number) => pad + (i / (n - 1)) * (W - 2 * pad);
  const minM = Math.min(...items.map((d) => d.metric));
  const maxM = Math.max(...items.map((d) => d.metric));

  return (
    <div className="rounded-xl bg-[rgba(10,14,23,0.6)] p-5 flex flex-col gap-4">
      {/* Wikipedia 定义 */}
      <a
        href={wikiUrl}
        target="_blank"
        rel="noreferrer"
        className="rounded-lg p-3 flex gap-2 items-start bg-[rgba(66,133,244,0.08)] border border-[#4285f4]/30 hover:bg-[rgba(66,133,244,0.14)] transition-all"
      >
        <BookMarked className="w-4 h-4 mt-0.5 text-[#4285f4] shrink-0" />
        <div>
          <div className="text-[11px] text-[#4285f4] font-semibold mb-0.5">权威定义 · Wikipedia（点击查看原文 →）</div>
          <div className="text-sm text-slate-300 leading-relaxed">{wikiDef}</div>
        </div>
      </a>

      {/* 时间轴 + mAP 折线 */}
      <svg viewBox={`0 0 ${W} 150`} className="w-full">
        {/* mAP 趋势折线 */}
        <polyline
          points={items.map((d, i) => `${x(i)},${120 - ((d.metric - minM) / (maxM - minM || 1)) * 80}`).join(" ")}
          fill="none"
          stroke="#00e5ff"
          strokeWidth={1.5}
          opacity={0.5}
        />
        {/* 主轴 */}
        <line x1={pad} y1={128} x2={W - pad} y2={128} stroke="#334155" strokeWidth={2} />
        {items.map((d, i) => {
          const active = i === current;
          const my = 120 - ((d.metric - minM) / (maxM - minM || 1)) * 80;
          return (
            <g key={i}>
              <circle cx={x(i)} cy={my} r={active ? 5 : 3} fill={active ? "#00ff88" : "#00e5ff"} opacity={active ? 1 : 0.5} />
              <circle cx={x(i)} cy={128} r={active ? 6 : 3} fill={active ? "#00ff88" : "#475569"} />
              <text x={x(i)} y={145} textAnchor="middle" fill={active ? "#00ff88" : "#64748b"} fontSize={9} className="font-mono">
                {d.year}
              </text>
              {active && (
                <text x={x(i)} y={my - 10} textAnchor="middle" fill="#00ff88" fontSize={11} className="font-mono">
                  {d.metric}
                </text>
              )}
            </g>
          );
        })}
      </svg>

      {/* 当前版本详情 */}
      <div className="glass rounded-lg p-4 border border-[#00ff88]/20">
        <div className="flex items-center gap-2 flex-wrap mb-2">
          <span className="text-lg font-bold text-[#00ff88]">{cur.name}</span>
          {cur.tag && (
            <span className="text-[11px] font-mono px-1.5 py-0.5 rounded bg-[rgba(179,136,255,0.15)] text-[#b388ff] border border-[#b388ff]/30">
              {cur.tag}
            </span>
          )}
        </div>
        <div className="flex flex-wrap gap-x-6 gap-y-1 text-xs text-slate-400 mb-2">
          <span className="flex items-center gap-1"><Calendar className="w-3 h-3" /> {cur.year}</span>
          <span className="flex items-center gap-1"><Users className="w-3 h-3" /> {cur.org}</span>
          <span className="flex items-center gap-1 text-[#00e5ff]">COCO mAP ≈ {cur.metric}</span>
        </div>
        <div className="flex gap-2 items-start text-sm text-slate-300 leading-relaxed">
          <Lightbulb className="w-3.5 h-3.5 mt-0.5 text-[#ffab40] shrink-0" />
          {cur.innovation}
        </div>
      </div>
    </div>
  );
}
