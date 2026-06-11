// 位置编码的纯数学（供 builder/Viz 复用、单测覆盖）。

export function dot(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

// 正弦位置编码：PE(pos,2i)=sin(pos/10000^(2i/d))，PE(pos,2i+1)=cos(...)
export function sinusoidalPE(pos: number, dim: number): number[] {
  const v: number[] = [];
  for (let i = 0; i < dim; i += 2) {
    const freq = 1 / Math.pow(10000, i / dim);
    v.push(Math.sin(pos * freq));
    if (v.length < dim) v.push(Math.cos(pos * freq));
  }
  return v.slice(0, dim);
}

// RoPE：把相邻 2D 对按角度 pos*freq_i 旋转。
export function ropeApply(vec: number[], pos: number, base = 10000): number[] {
  const d = vec.length;
  const out = vec.slice();
  for (let i = 0; i + 1 < d; i += 2) {
    const freq = 1 / Math.pow(base, i / d);
    const theta = pos * freq;
    const c = Math.cos(theta);
    const s = Math.sin(theta);
    const x = vec[i];
    const y = vec[i + 1];
    out[i] = x * c - y * s;
    out[i + 1] = x * s + y * c;
  }
  return out;
}

// 单个 2D 对在某位置的旋转角（给 Viz 画 angle-vs-pos 曲线用）。
export function ropeAngle(pairIndex: number, pos: number, dim: number, base = 10000): number {
  const freq = 1 / Math.pow(base, (2 * pairIndex) / dim);
  return pos * freq;
}

// YaRN：用更大的 base 拉长波长→旋转更慢→有效上下文更长（NTK-aware 的简化演示）。
export function yarnApply(vec: number[], pos: number, scale = 8, base = 10000): number[] {
  return ropeApply(vec, pos, base * scale);
}
export function yarnAngle(pairIndex: number, pos: number, dim: number, scale = 8, base = 10000): number {
  return ropeAngle(pairIndex, pos, dim, base * scale);
}
