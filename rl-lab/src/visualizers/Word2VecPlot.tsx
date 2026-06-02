// embedding 家族（Word2Vec）：2D 词向量散点 + 词标签，按语义组着色。
import { EmbeddingState } from "@/player/types";
import { makeScale, CLUSTER_COLORS } from "./plot";

const W = 560;
const H = 380;
const M = 40;

export default function Word2VecPlot({ state }: { state: EmbeddingState }) {
  const { words, positions, groups } = state;
  const s = makeScale(positions, W, H, M);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full rounded-xl bg-[rgba(10,14,23,0.6)]">
      <rect x={M} y={M} width={W - 2 * M} height={H - 2 * M} fill="none" stroke="rgba(0,255,136,0.12)" />
      {positions.map((p, i) => {
        const col = CLUSTER_COLORS[groups[i] % CLUSTER_COLORS.length];
        return (
          <g key={i}>
            <circle cx={s.sx(p.x)} cy={s.sy(p.y)} r={4} fill={col} />
            <text x={s.sx(p.x) + 7} y={s.sy(p.y) + 4} fill={col} fontSize={12} className="font-mono">
              {words[i]}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
