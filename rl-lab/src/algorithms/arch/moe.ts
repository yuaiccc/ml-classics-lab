import { Trajectory } from "@/player/types";
import { MoEState } from "@/visualizers/arch/types";

export function runMoE(): Trajectory<MoEState> {
  const tokens = [1, 4, 7, 2, 9, 5, 3, 8, 6, 0, 11, 13];
  return {
    meta: {
      id: "moe",
      title: "专家混合 MoE · Dense→MoE→Switch→DeepSeekMoE",
      family: "moe",
      algorithm: "Mixture of Experts",
      description: "如何把模型参数堆大，但每个 token 只算一小部分？",
      tutorial: {
        problem: "想要更大的模型容量，又不想让每个 token 都过一遍全部参数（太贵）。",
        intuition:
          "MoE 把 FFN 拆成很多「专家」，一个路由器给每个 token 只挑少数几个专家来算——参数总量大、单 token 计算量小（稀疏激活）。Switch 极端到每 token 只选 1 个专家。DeepSeekMoE 用更细粒度的专家 + 几个常驻「共享专家」，兼顾专精与通用。难点是负载均衡：别让少数专家被挤爆、其余闲置。",
        watch: [
          "切 Dense→MoE→Switch：每个 token 点亮的专家越来越少（越稀疏）",
          "稀疏度条：激活专家/总专家，Dense=100%，Switch=1/N",
          "负载均衡条：各专家被选中的次数，越均匀越好",
          "DeepSeekMoE：常驻共享专家始终点亮 + 细粒度路由",
        ],
        concepts: [
          { term: "专家 Expert", explain: "一个独立的 FFN 子网络" },
          { term: "路由器 Router", explain: "给每个 token 打分、挑选 top-k 专家" },
          { term: "稀疏激活", explain: "总参数大，但每 token 只算少数专家" },
          { term: "负载均衡", explain: "让 token 均匀分到各专家，避免少数被挤爆" },
        ],
        tryThis: "拖动 token 滑块逐个看路由；切到 Switch 看是否只点亮 1 个专家。",
      },
    },
    frames: [{ iter: 0, state: { tokens, nExperts: 8, topK: 2, nShared: 2 }, metrics: {} }],
  };
}
