import { KVVariant, AttnKVConfig } from "@/visualizers/arch/types";

// 各变体的 KV 头数（MLA 用压缩潜向量，不以「头数」计，返回 1 仅作占位）。
export function kvHeads(variant: KVVariant, cfg: AttnKVConfig): number {
  switch (variant) {
    case "mha": return cfg.nHeads;
    case "mqa": return 1;
    case "gqa": return cfg.nGroups;
    case "mla": return 1;
  }
}

// 每个 token 需缓存的元素数：K、V 各占 kvHeads*dHead；MLA 只缓存一个 latentDim 向量。
export function cacheElemsPerToken(variant: KVVariant, cfg: AttnKVConfig): number {
  if (variant === "mla") return cfg.latentDim;
  return 2 * kvHeads(variant, cfg) * cfg.dHead;
}

// 整条序列的 KV-cache 总元素数。
export function cacheTotal(variant: KVVariant, cfg: AttnKVConfig): number {
  return cfg.seqLen * cacheElemsPerToken(variant, cfg);
}
