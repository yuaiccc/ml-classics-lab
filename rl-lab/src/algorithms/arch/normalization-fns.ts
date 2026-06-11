// 归一化的纯数学（供 Viz 复用、单测覆盖）。
export function mean(v: number[]): number {
  return v.reduce((a, b) => a + b, 0) / v.length;
}
export function rms(v: number[]): number {
  return Math.sqrt(v.reduce((a, b) => a + b * b, 0) / v.length);
}
export function cosine(a: number[], b: number[]): number {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-12);
}

// LayerNorm：去均值再除以标准差（中心化 + 缩放）。
export function layerNorm(v: number[], eps = 1e-5): number[] {
  const m = mean(v);
  const variance = v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length;
  const denom = Math.sqrt(variance + eps);
  return v.map((x) => (x - m) / denom);
}

// RMSNorm：只按均方根缩放，不去均值（保持方向）。
export function rmsNorm(v: number[], eps = 1e-5): number[] {
  const ms = v.reduce((a, b) => a + b * b, 0) / v.length;
  const denom = Math.sqrt(ms + eps);
  return v.map((x) => x / denom);
}
