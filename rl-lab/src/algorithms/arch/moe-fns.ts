import type { MoEVariant } from "@/visualizers/arch/types";

// 确定性门控分数：用 token 值与专家索引算一个 [0,1] 分数。
export function gateScores(token: number, nExperts: number): number[] {
  const s: number[] = [];
  for (let e = 0; e < nExperts; e++) {
    s.push(Math.sin(token * 0.7 + e * 1.3) * 0.5 + 0.5);
  }
  return s;
}

// 分数最高的 k 个专家索引（降序）。
export function topKExperts(scores: number[], k: number): number[] {
  return scores
    .map((v, i) => [v, i] as [number, number])
    .sort((a, b) => b[0] - a[0])
    .slice(0, Math.max(0, k))
    .map((x) => x[1]);
}

// 该 token 经路由激活的专家索引（不含常驻共享专家，由 Viz 单独标）。
export function routedExperts(variant: MoEVariant, token: number, nExperts: number, topK: number): number[] {
  const scores = gateScores(token, nExperts);
  if (variant === "dense") return scores.map((_, i) => i);
  if (variant === "switch") return topKExperts(scores, 1);
  return topKExperts(scores, topK); // moe / deepseek
}

// 每个 token 激活的专家占比（稀疏度）。
export function activeFraction(variant: MoEVariant, nExperts: number, topK: number, nShared: number): number {
  if (variant === "dense") return 1;
  if (variant === "switch") return 1 / nExperts;
  if (variant === "moe") return topK / nExperts;
  return (topK + nShared) / (nExperts + nShared); // deepseek
}
