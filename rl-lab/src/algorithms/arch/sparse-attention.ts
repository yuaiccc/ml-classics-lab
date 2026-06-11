import { Trajectory } from "@/player/types";
import { SparseAttnState } from "@/visualizers/arch/types";

export function runSparseAttention(): Trajectory<SparseAttnState> {
  return {
    meta: {
      id: "sparse-attention",
      title: "稀疏·窗口注意力 · Full/Sparse/SWA/DSA/CSA",
      family: "sparse-attn",
      algorithm: "Sparse & Windowed Attention",
      description: "全注意力是 O(L²)。只算一部分格子能省多少、又保留什么？",
      tutorial: {
        problem: "全连接注意力要算 L×L 个格子，长序列代价爆炸。能不能只算「重要的」格子？",
        intuition:
          "稀疏/窗口注意力按某种图案只计算一部分注意力格子：滑动窗口（SWA）只看最近若干个 token；Sparse Transformer 用局部窗口 + 固定全局列；DSA 动态地选要算哪些；CSA/HCA 压缩或跨步。代价小很多、效果接近全注意力——因为真实注意力本来就很「集中」（看右边真实样本）。",
        watch: [
          "切变体看掩码图案：哪些格子被计算（亮）",
          "FLOPs 占比：被算格子 / 全矩阵，越稀疏越省",
          "调窗口大小，看 SWA 的带宽变化",
          "「真实样本」：本地真训 Transformer 的真实注意力——天然集中/稀疏",
        ],
        concepts: [
          { term: "因果掩码", explain: "每个 token 只能注意自己左边（含自己）" },
          { term: "SWA 滑动窗口", explain: "只注意最近 window 个 token（Longformer/Mistral）" },
          { term: "Sparse Transformer", explain: "局部窗口 + 固定全局位置的稀疏图案" },
          { term: "DSA/CSA", explain: "动态稀疏 / 压缩注意力（较新，按公开描述近似演示）" },
        ],
        tryThis: "把序列调长、切到 SWA，看 FLOPs 占比怎么从 ~50% 掉到很小。",
      },
    },
    frames: [{ iter: 0, state: { seqLen: 24 }, metrics: {} }],
  };
}
