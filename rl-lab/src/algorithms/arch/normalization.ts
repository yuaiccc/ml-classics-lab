import { Trajectory } from "@/player/types";
import { NormState } from "@/visualizers/arch/types";

export function runNormalization(): Trajectory<NormState> {
  // 固定的玩具激活向量（d=3，均值/尺度各异，便于看清中心化与缩放）。
  const vectors = [
    [2, 1, -1],
    [3, 3, 0],
    [-1, -2, -3],
    [0.5, 2, 4],
    [5, -1, 2],
    [1, 1, 1],
  ];
  return {
    meta: {
      id: "normalization",
      title: "归一化 · LayerNorm→RMSNorm→QKNorm→Pre/Post",
      family: "normalization",
      algorithm: "Normalization",
      description: "深层网络靠归一化才训得稳，它们到底做了什么、差别在哪？",
      tutorial: {
        problem: "层一深，激活值的尺度就乱飘，梯度爆炸/消失。归一化把每层激活拉回稳定范围。",
        intuition:
          "LayerNorm 先去均值再除以标准差（中心化+缩放）。RMSNorm 省掉去均值，只按均方根缩放——更快、且现代大模型（LLaMA 等）证明效果不输。QKNorm 把注意力里的 Q/K 先归一化，压住过大的注意力 logits。Pre-Norm（归一化放在残差之前）让深层训练更稳，是现代 Transformer 的默认；Post-Norm 表达力强但难训。",
        watch: [
          "LayerNorm：每个向量归一化后均值≈0、与原方向不同",
          "RMSNorm：只缩放、方向不变（余弦相似度≈1），且不强行去均值",
          "QKNorm：归一化 Q/K 后注意力 logits 的幅度被压住",
          "Pre vs Post：残差流幅度随层深的增长曲线，Post 更易发散",
        ],
        concepts: [
          { term: "LayerNorm", explain: "(x−均值)/标准差，中心化+缩放" },
          { term: "RMSNorm", explain: "x/RMS，只缩放不去均值，保持方向、更省算力" },
          { term: "QKNorm", explain: "对 Q/K 归一化以稳定注意力分数" },
          { term: "Pre/Post-Norm", explain: "归一化放残差前(Pre,更稳)还是后(Post,更强但难训)" },
        ],
        tryThis: "在 LayerNorm 和 RMSNorm 间切换，盯住每个向量的「方向(余弦)」是否改变。",
      },
    },
    frames: [{ iter: 0, state: { vectors }, metrics: {} }],
  };
}
