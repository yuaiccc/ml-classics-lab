// 线性/SSM 序列模型的核心对比：固定大小的递归状态 vs 注意力随长度增长的 KV-cache。

// 注意力：KV-cache 随序列长度线性增长（每个历史 token 都要缓存）。
export function attentionMemory(L: number): number {
  return L;
}
// 线性/SSM：只维护一个固定大小的递归状态，与序列长度无关。
export function recurrentMemory(stateSize: number): number {
  return stateSize;
}

// 门控线性递归一步：s' = decay*s + gate*x（逐元素）。
export function scanStep(state: number[], x: number[], decay: number, gate: number): number[] {
  return state.map((s, i) => decay * s + gate * (x[i] ?? 0));
}

// 把标量 token 序列喂进一个 stateSize 维递归状态，返回每步状态的范数（给 Viz 画扫描）。
export function scanNorms(tokens: number[], stateSize: number, decay = 0.8, gate = 0.5): number[] {
  let state = new Array(stateSize).fill(0);
  const norms: number[] = [];
  for (const t of tokens) {
    const x = new Array(stateSize).fill(0).map((_, i) => Math.sin(t * 0.5 + i));
    state = scanStep(state, x, decay, gate);
    norms.push(Math.sqrt(state.reduce((a, b) => a + b * b, 0)));
  }
  return norms;
}
