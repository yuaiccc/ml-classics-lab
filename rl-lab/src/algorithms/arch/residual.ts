import { Trajectory } from "@/player/types";
import { ResidualState } from "@/visualizers/arch/types";

export function runResidual(): Trajectory<ResidualState> {
  return {
    meta: {
      id: "residual",
      title: "残差连接 · Plain→ResNet→HC→mHC→AttnResidual",
      family: "residual",
      algorithm: "Residual Connections",
      description: "网络一深就训不动，残差连接如何让梯度活下来？",
      tutorial: {
        problem: "层一多，反向传播的梯度按层数指数衰减/爆炸，深层网络根本训不动。",
        intuition:
          "残差连接（ResNet 的 x+f(x)）给梯度开一条「恒等捷径」：就算 f 这条路梯度很小，那个「+1」也能把梯度直通回去，于是几百层也训得动。它是 Transformer 能堆深的前提。HC/mHC 用多条并行残差流提升表达与稳定，AttnResidual 用注意力来调制残差流。",
        watch: [
          "Plain：梯度幅度随离输出越远指数衰减→早层学不到（消失）",
          "RC：恒等捷径让梯度≥1，越深越稳",
          "HC/mHC：多条并行残差流，进一步抬高/稳住梯度",
          "右侧示意图：skip 连接（捷径）怎么绕过 f",
        ],
        concepts: [
          { term: "残差连接", explain: "y = x + f(x)，给梯度一条恒等直通路" },
          { term: "梯度消失", explain: "无捷径时梯度按层数指数衰减，深层学不到" },
          { term: "HC/mHC", explain: "Hyper-Connections：多条并行残差流" },
          { term: "AttnResidual", explain: "用注意力调制残差流的较新变体" },
        ],
        tryThis: "切到 Plain 看梯度曲线贴地（消失），再切回 RC 看它被「+1」托起来。",
      },
    },
    frames: [{ iter: 0, state: { nLayers: 24 }, metrics: {} }],
  };
}
