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

// —— 位置编码 ——
export type PosEncVariant = "none" | "sinusoidal" | "rope" | "nope" | "yarn";
export interface PosEncState {
  dim: number; // 演示维度（如 32）
  maxPos: number; // 位置轴长度（如 64）
}

// —— 归一化 ——
export type NormVariant = "layernorm" | "rmsnorm" | "qknorm" | "prepost";
export interface NormState {
  vectors: number[][]; // 若干 d 维玩具激活向量（这里 d=3）
}

// —— MoE ——
export type MoEVariant = "dense" | "moe" | "switch" | "deepseek";
export interface MoEState {
  tokens: number[]; // 玩具 token 序列（值用于确定性路由）
  nExperts: number;
  topK: number;
  nShared: number; // DeepSeekMoE 常驻共享专家数
}

// —— 残差连接 ——
export type ResidualVariant = "plain" | "rc" | "hc" | "mhc" | "attnres";
export interface ResidualState {
  nLayers: number;
}

// —— 线性·SSM ——
export type LinearSeqVariant = "lightning" | "gateddeltanet" | "kda" | "mamba" | "mamba2" | "mamba3";
export interface LinearSeqState {
  tokens: number[];
  stateSize: number;
}

// —— 稀疏·窗口注意力 ——
export type SparseAttnVariant = "full" | "sparse" | "swa" | "dsa" | "csa";
export interface SparseAttnState {
  seqLen: number;
}
