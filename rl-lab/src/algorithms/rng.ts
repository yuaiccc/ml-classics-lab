// 可复现的伪随机数：同一个 seed 产生同一组数据，方便“重新生成”按钮换 seed。

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box-Muller：从均匀分布生成高斯噪声 */
export function gaussian(rng: () => number, mean = 0, std = 1): number {
  const u = 1 - rng();
  const v = rng();
  return mean + std * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
