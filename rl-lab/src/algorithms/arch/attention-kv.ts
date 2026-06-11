import { Trajectory } from "@/player/types";
import { AttnKVState } from "@/visualizers/arch/types";

export function runAttentionKV(): Trajectory<AttnKVState> {
  return {
    meta: {
      id: "attention-kv",
      title: "注意力 KV 变体 · MHA→MQA→GQA→MLA",
      family: "attn-kv",
      algorithm: "Attention KV sharing",
      description: "推理时 KV-cache 是显存大头。如何在不太掉效果下把它缩小？",
      tutorial: {
        problem: "自回归推理要缓存每个 token 的 K/V（KV-cache），长上下文时显存爆炸。",
        intuition:
          "MHA 每个查询头都有自己的一份 K/V，最准但最占显存。MQA 让所有头共享 1 份 K/V，省到极致但可能掉点。GQA 折中：分几组、组内共享。MLA（DeepSeek）更进一步：把 K/V 压成一个低秩潜向量缓存、用时再上投，显存最省且尽量保效果。",
        watch: [
          "切换 MHA/MQA/GQA/MLA，看 Q 头与 KV 头的连线如何「合并共享」",
          "右侧 KV-cache 显存条：从 MHA 到 MLA 一路缩小",
          "拖动头数/分组数，观察 GQA 在 MHA 与 MQA 之间连续过渡",
        ],
        concepts: [
          { term: "KV-cache", explain: "自回归推理缓存的历史 K/V，长度随上下文线性增长" },
          { term: "MHA", explain: "Multi-Head：每个 Q 头独立 K/V，质量高、显存大" },
          { term: "MQA", explain: "Multi-Query：所有 Q 头共享 1 份 K/V，显存最省" },
          { term: "GQA", explain: "Grouped-Query：分组共享 K/V，质量/显存折中" },
          { term: "MLA", explain: "Multi-head Latent：K/V 压成低秩潜向量缓存，再上投复原" },
        ],
        tryThis: "把分组数从 1 调到 8，看 GQA 如何在 MQA 和 MHA 两端之间滑动。",
      },
    },
    frames: [
      {
        iter: 0,
        state: { config: { nHeads: 8, nGroups: 2, dHead: 64, seqLen: 1024, latentDim: 64 } },
        metrics: {},
      },
    ],
  };
}
