// 「LLM 架构解剖」板块各实验的 state 形状（按需逐批扩充）。

// —— 激活函数 ——
export type ActivationVariant = "relu" | "gelu" | "silu" | "swish" | "swiglu";
export interface ActivationState {
  xs: number[]; // 采样横坐标（如 [-6,6] 等距 241 点）
}

// —— 注意力 KV 变体 ——
export type KVVariant = "mha" | "mqa" | "gqa" | "mla";
export interface AttnKVConfig {
  nHeads: number; // Q 头数
  nGroups: number; // GQA 分组数（KV 头数）
  dHead: number; // 每头维度
  seqLen: number; // 序列长度（算 KV-cache）
  latentDim: number; // MLA 压缩潜维
}
export interface AttnKVState {
  config: AttnKVConfig;
}
