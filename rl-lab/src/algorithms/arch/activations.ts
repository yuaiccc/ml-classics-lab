import { Trajectory } from "@/player/types";
import { ActivationState } from "@/visualizers/arch/types";

export function runActivations(): Trajectory<ActivationState> {
  const N = 241;
  const xs: number[] = [];
  for (let i = 0; i < N; i++) xs.push(-6 + (12 * i) / (N - 1));
  return {
    meta: {
      id: "activations",
      title: "激活函数 · ReLU→GELU→SiLU→Swish→SwiGLU",
      family: "activation",
      algorithm: "Activations",
      description: "现代 LLM 的 FFN 用什么激活？对比曲线、导数与门控。",
      tutorial: {
        problem: "神经网络要非线性才能拟合复杂函数——用哪个非线性？",
        intuition:
          "ReLU 简单但负区死掉（梯度为 0）。GELU/SiLU/Swish 在 0 附近平滑、负区有轻微下凹，训练更稳、表达更强。SwiGLU 把激活做成「门控」：一路当开关乘到另一路上，是现代 LLM（LLaMA 等）FFN 的主流。",
        watch: [
          "切换变体，看曲线在负区的差别：ReLU 硬截断，其余平滑下凹",
          "看导数曲线：ReLU 导数是阶跃，平滑激活的导数连续",
          "拖动 Swish 的 β：β→0 趋近线性，β→大 趋近 ReLU",
          "SwiGLU 面板：门控如何把一路压制/放大另一路",
        ],
        concepts: [
          { term: "ReLU", explain: "max(0,x)，最简单的非线性，负区梯度为 0" },
          { term: "SiLU/Swish", explain: "x·sigmoid(βx)，平滑、负区有下凹，β=1 时 SiLU=Swish" },
          { term: "GELU", explain: "用高斯 CDF 加权输入，BERT/GPT 常用" },
          { term: "SwiGLU", explain: "门控线性单元：SiLU(xW)·(xV)，现代 LLM FFN 主流" },
        ],
        tryThis: "把 β 从 1 调到 5，观察 Swish 怎样越来越像 ReLU。",
      },
    },
    frames: [{ iter: 0, state: { xs }, metrics: {} }],
  };
}
