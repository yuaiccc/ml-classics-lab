import { Trajectory } from "@/player/types";
import { LinearSeqState } from "@/visualizers/arch/types";

export function runLinearSeq(): Trajectory<LinearSeqState> {
  const tokens = [1, 4, 7, 2, 9, 5, 3, 8, 6, 0, 2, 7, 4, 1, 9];
  return {
    meta: {
      id: "linear-seq",
      title: "线性·SSM · Lightning/DeltaNet/KDA/Mamba 1-3",
      family: "linear-seq",
      algorithm: "Linear-time Sequence Mixing",
      description: "注意力是 O(L²)、KV-cache 随长度膨胀。线性/SSM 怎么做到 O(L)、定长状态？",
      tutorial: {
        problem: "注意力处理长序列代价是 O(L²)，推理时 KV-cache 还随长度线性膨胀，长上下文吃不消。",
        intuition:
          "线性注意力与状态空间模型（SSM）把「看所有历史」改成「维护一个固定大小的递归状态」：每来一个 token 就更新状态，不用缓存全部历史——内存与序列长度无关、计算线性。Lightning 是线性注意力，GatedDeltaNet/KDA 加了门控与 delta 更新规则，Mamba 系用输入相关的选择性 SSM。代价是「压缩历史」可能丢细节，各变体都在如何更聪明地记忆上做文章。",
        watch: [
          "内存对比曲线：注意力随长度直线上升，线性/SSM 一条水平线（定长状态）",
          "扫描视图：固定大小的状态随 token 流逐步更新",
          "切变体看说明：各自改了递归/门控的什么",
        ],
        concepts: [
          { term: "KV-cache", explain: "注意力要缓存全部历史 K/V，内存 ∝ 序列长度" },
          { term: "递归状态", explain: "SSM/线性注意力只维护定长状态，内存与长度无关" },
          { term: "选择性 SSM", explain: "Mamba：让状态更新依赖输入内容，选择性记忆" },
          { term: "delta 规则", explain: "DeltaNet/KDA：用误差驱动的状态更新提升记忆精度" },
        ],
        tryThis: "拉长序列长度，看注意力内存直线飙升而 SSM 纹丝不动。",
      },
    },
    frames: [{ iter: 0, state: { tokens, stateSize: 16 }, metrics: {} }],
  };
}
