// 每个实验涉及的参考论文（用户提供，原样挂链接）。
// note 用于标注「按公开描述近似演示」的极新/未公开论文。
export interface PaperRef {
  name: string;
  url: string;
  note?: string;
}

export const PAPERS: Record<string, PaperRef[]> = {
  activations: [
    { name: "GELU", url: "https://arxiv.org/abs/1606.08415" },
    { name: "SiLU", url: "https://arxiv.org/abs/1702.03118" },
    { name: "Swish", url: "https://arxiv.org/abs/1710.05941v1" },
  ],
  "attention-kv": [
    { name: "MHA (Attention Is All You Need)", url: "https://arxiv.org/abs/1706.03762v7" },
    { name: "MQA", url: "https://arxiv.org/abs/1911.02150" },
    { name: "GQA", url: "https://arxiv.org/abs/2305.13245" },
    { name: "MLA (DeepSeek-V2)", url: "https://arxiv.org/abs/2405.04434" },
  ],
  "pos-encoding": [
    { name: "正弦 PE", url: "https://arxiv.org/abs/1706.03762v7" },
    { name: "RoPE", url: "https://arxiv.org/abs/2104.09864" },
    { name: "NoPE", url: "https://arxiv.org/abs/2305.19466" },
    { name: "YaRN", url: "https://arxiv.org/abs/2309.00071" },
  ],
  normalization: [
    { name: "LayerNorm", url: "https://arxiv.org/abs/1607.06450" },
    { name: "RMSNorm", url: "https://arxiv.org/abs/1910.07467" },
    { name: "QK/KVNorm", url: "https://arxiv.org/abs/2010.04245" },
    { name: "Pre/Post-Norm", url: "https://arxiv.org/abs/2002.04745" },
  ],
  "sparse-attention": [
    { name: "Sparse Transformer", url: "https://arxiv.org/abs/1904.10509" },
    { name: "SWA (Longformer)", url: "https://arxiv.org/abs/2004.05150" },
    { name: "Gemma2 SWA", url: "https://arxiv.org/abs/2408.00118" },
    { name: "DSA", url: "https://arxiv.org/abs/2512.02556", note: "极新论文，按公开描述近似演示" },
    { name: "CSA/HCA (DeepSeek-V4)", url: "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf", note: "未公开，按公开描述近似演示" },
  ],
  "linear-seq": [
    { name: "Lightning Attention", url: "https://arxiv.org/abs/2401.04658" },
    { name: "Gated DeltaNet", url: "https://arxiv.org/abs/2412.06464" },
    { name: "KDA", url: "https://arxiv.org/abs/2510.26692" },
    { name: "Mamba", url: "https://arxiv.org/abs/2312.00752" },
    { name: "Mamba-2", url: "https://arxiv.org/abs/2405.21060" },
    { name: "Mamba-3", url: "https://arxiv.org/abs/2603.15569", note: "极新论文，按公开描述近似演示" },
  ],
  moe: [
    { name: "MoE (Sparsely-Gated)", url: "https://arxiv.org/abs/1701.06538" },
    { name: "Switch Transformer", url: "https://arxiv.org/abs/2101.03961" },
    { name: "DeepSeekMoE", url: "https://arxiv.org/abs/2401.06066" },
  ],
  residual: [
    { name: "RC (ResNet)", url: "https://arxiv.org/abs/1512.03385" },
    { name: "HC", url: "https://arxiv.org/abs/2409.19606" },
    { name: "mHC", url: "https://arxiv.org/abs/2512.24880", note: "极新论文，按公开描述近似演示" },
    { name: "AttnResidual", url: "https://arxiv.org/abs/2603.15031", note: "极新论文，按公开描述近似演示" },
  ],
};
