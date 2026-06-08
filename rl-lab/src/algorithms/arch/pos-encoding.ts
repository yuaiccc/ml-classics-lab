import { Trajectory } from "@/player/types";
import { PosEncState } from "@/visualizers/arch/types";

export function runPosEncoding(): Trajectory<PosEncState> {
  return {
    meta: {
      id: "pos-encoding",
      title: "位置编码 · PE→RoPE→NoPE→YaRN",
      family: "pos-encoding",
      algorithm: "Positional Encoding",
      description: "注意力本身对顺序无感，位置信息怎么注入？",
      tutorial: {
        problem: "自注意力是「集合」运算，打乱 token 顺序结果不变——可语言是有序的。位置信息怎么加进去？",
        intuition:
          "正弦 PE 给每个位置一串固定的 sin/cos 指纹，直接加到词向量上。RoPE 改成「旋转」：按位置把 Q/K 的二维对转一个角度，巧妙之处是注意力分数 q·k 只依赖两者的相对距离。NoPE 干脆不加，靠因果掩码隐式获得位置感。YaRN 在 RoPE 基础上把旋转频率调慢，让训练在短上下文、推理能外推到长上下文。",
        watch: [
          "正弦 PE：看位置×维度的热力图条纹，高维变化慢、低维变化快",
          "RoPE：拖位置滑块，看二维查询向量被旋转；分数只看相对距离",
          "YaRN：对比 RoPE 与 YaRN 的角度-位置曲线，YaRN 转得更慢→能外推更远",
          "NoPE：完全不加位置，靠因果掩码隐式排序",
        ],
        concepts: [
          { term: "正弦 PE", explain: "不同频率的 sin/cos 给每个位置编码，加到词向量上" },
          { term: "RoPE", explain: "按位置旋转 Q/K，使注意力分数只依赖相对位置" },
          { term: "NoPE", explain: "不显式加位置编码，靠因果掩码获得顺序信息" },
          { term: "YaRN", explain: "调慢 RoPE 频率以把短上下文模型外推到长上下文" },
        ],
        tryThis: "在 RoPE 下拖动位置滑块，观察向量匀速旋转；切到 YaRN 看曲线变缓。",
      },
    },
    frames: [{ iter: 0, state: { dim: 32, maxPos: 64 }, metrics: {} }],
  };
}
