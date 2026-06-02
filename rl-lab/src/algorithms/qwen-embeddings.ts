// 本地 Qwen 真实语义嵌入 —— 大模型怎么"理解"词义。
// qwen3-embedding 把每个词转成 1024 维向量（已离线算好相似度矩阵），
// 浏览器端用 MDS 把高维相似度还原成 2D：随机初始 → 漂成语义簇。
// 这是玩具版 Word2Vec 的"真·大模型版"——同义/同类词向量天然靠近。
import { Trajectory, EmbeddingState, Frame } from "@/player/types";
import { mulberry32 } from "./rng";
import qwen from "@/data/qwen-embeddings.json";

export interface QwenEmbOptions {
  seed?: number;
  iters?: number;
  lr?: number;
}

export function runQwenEmbeddings(opts: QwenEmbOptions = {}): Trajectory<EmbeddingState> {
  const { seed = (Date.now() & 0xffff) >>> 0, iters = 90, lr = 0.1 } = opts;
  const words = qwen.words as string[];
  const groups = qwen.groups as number[];
  const sim = qwen.sim as number[][];
  const n = words.length;
  const rng = mulberry32(seed);

  // 目标距离：单位向量间 ||a-b|| = sqrt(2-2cos)
  const D = sim.map((row) => row.map((s) => Math.sqrt(Math.max(0, 2 - 2 * s))));

  const pos = Array.from({ length: n }, () => ({ x: (rng() * 2 - 1) * 0.6, y: (rng() * 2 - 1) * 0.6 }));

  const stress = () => {
    let s = 0;
    for (let i = 0; i < n; i++)
      for (let j = i + 1; j < n; j++) {
        const d = Math.hypot(pos[i].x - pos[j].x, pos[i].y - pos[j].y);
        s += (d - D[i][j]) ** 2;
      }
    return s;
  };

  const frames: Frame<EmbeddingState>[] = [];
  const snap = (it: number) =>
    frames.push({
      iter: it,
      state: { words, positions: pos.map((p) => ({ ...p })), groups },
      metrics: { stress: stress() },
    });
  snap(0);

  // MDS：梯度下降最小化 Σ(实际距离 - 目标距离)²
  for (let t = 1; t <= iters; t++) {
    const grad = pos.map(() => ({ x: 0, y: 0 }));
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        const dx = pos[i].x - pos[j].x;
        const dy = pos[i].y - pos[j].y;
        const dist = Math.hypot(dx, dy) || 1e-6;
        const c = (2 * (dist - D[i][j])) / dist;
        grad[i].x += c * dx;
        grad[i].y += c * dy;
      }
    for (let i = 0; i < n; i++) {
      pos[i].x -= (lr * grad[i].x) / n;
      pos[i].y -= (lr * grad[i].y) / n;
    }
    if (t % 2 === 0) snap(t);
  }

  return { meta: QWEN_EMB_META, frames };
}

export const QWEN_EMB_META = {
  id: "qwen-embeddings",
  title: "本地 Qwen · 语义嵌入",
  family: "embedding" as const,
  algorithm: "Qwen Embeddings + MDS",
  description: "用你本地 Ollama 的 qwen3-embedding 把词转成 1024 维向量，MDS 还原成 2D，同义词自动聚成簇。",
  tutorial: {
    problem: "大模型怎么“理解”词义？它把每个词编码成一个高维向量，含义相近的词向量也相近。",
    intuition:
      "这里的向量是你本地 Ollama 里的 qwen3-embedding 真实算出来的（1024 维），不是玩具数据。算好两两词的相似度后，浏览器端用 MDS（多维缩放）把这种高维相似关系压到 2D：随机撒开的词，逐步漂移到能保持原始相似度的位置——于是动物、国家、食物、情绪、颜色各自聚成一团。这正是 RAG / 语义搜索 / 大模型理解语言的基石。",
    watch: [
      "25 个词初始随机散布，颜色=语义类别（动物/国家/食物/情绪/颜色）",
      "随 MDS 迭代，同类词逐渐聚到一起——因为 Qwen 给它们的向量本来就近",
      "右侧 stress（布局误差）下降，说明 2D 越来越忠实地还原了高维语义",
    ],
    concepts: [
      { term: "词嵌入 Embedding", explain: "把词/句子编码成稠密向量，相近含义→相近向量" },
      { term: "余弦相似度", explain: "用向量夹角衡量语义相似，大模型检索/RAG 的核心" },
      { term: "MDS 多维缩放", explain: "把高维距离关系尽量忠实地压到低维以便可视化" },
    ],
    tryThis: "点上方「🔴 实时运行」用本地 Qwen 真的嵌入一遍；对比「Word2Vec」（玩具语料）聚类更干净。",
  },
  hyperparams: { source: "qwen3-embedding:0.6b", dim: 1024, words: 25 },
};
