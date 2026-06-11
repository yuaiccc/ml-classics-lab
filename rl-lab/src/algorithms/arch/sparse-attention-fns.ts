import type { SparseAttnVariant } from "@/visualizers/arch/types";

// 第 i 个 query 是否注意第 j 个 key（因果：只看 j<=i）。
export function attends(
  variant: SparseAttnVariant,
  i: number,
  j: number,
  L: number,
  window = 4,
  stride = 3,
): boolean {
  if (j > i) return false; // 因果掩码
  switch (variant) {
    case "full": return true;
    case "swa": return i - j < window; // 滑动窗口：只看最近 window 个
    case "sparse": return i - j < window || j % stride === 0; // 局部窗口 + 固定全局列
    case "dsa": return i - j < window || (i * 31 + j * 17) % 5 === 0; // 动态稀疏（伪随机近似演示）
    case "csa": return i - j < window || j % 2 === 0; // 压缩/跨步近似演示
  }
}

// 被计算的注意力格子占全矩阵的比例（FLOPs 代理）。
export function attendedFraction(
  variant: SparseAttnVariant,
  L: number,
  window = 4,
  stride = 3,
): number {
  let count = 0;
  for (let i = 0; i < L; i++) {
    for (let j = 0; j < L; j++) {
      if (attends(variant, i, j, L, window, stride)) count++;
    }
  }
  return count / (L * L);
}
