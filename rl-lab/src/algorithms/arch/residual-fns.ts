import type { ResidualVariant } from "@/visualizers/arch/types";

// 反向传播到第 depth 层时的梯度幅度（玩具模型）。depth: 输入端=0 … 输出端=nLayers。
// 直觉：plain 无捷径→梯度按 w 指数衰减（消失）或膨胀（爆炸）；
// 残差有恒等捷径，梯度至少有「+1」的直通路，不会消失。
export function gradMagnitude(
  variant: ResidualVariant,
  depth: number,
  nLayers: number,
  w = 0.8,
  streams = 1,
): number {
  const fromOutput = nLayers - depth;
  switch (variant) {
    case "plain": return Math.pow(w, fromOutput);
    case "rc": return 1 + fromOutput * 0.1;
    case "hc": return 1 + fromOutput * 0.1 * streams;
    case "mhc": return 1 + fromOutput * 0.12 * streams;
    case "attnres": return 1 + fromOutput * 0.08;
  }
}
