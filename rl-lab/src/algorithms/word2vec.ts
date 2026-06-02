// Word2Vec（skip-gram，2D 词向量）—— 浏览器手写。
// 在一个小语料上训练：经常一起出现的词，向量会越靠越近，于是同语义的词聚成一团。
// （著名的 king−man+woman≈queen 类比就来自这种向量空间。）
import { Trajectory, EmbeddingState, Frame } from "@/player/types";
import { mulberry32 } from "./rng";

const WORDS = ["king", "queen", "man", "woman", "paris", "france", "tokyo", "japan", "dog", "cat", "animal", "city"];
const GROUPS = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 1]; // 0=人/王室 1=地理 2=动物
const idx = (w: string) => WORDS.indexOf(w);

// 小语料：每个“句子”里的词彼此互为上下文
const SENTENCES = [
  ["king", "queen", "man", "woman"],
  ["king", "man"],
  ["queen", "woman"],
  ["paris", "france", "city"],
  ["tokyo", "japan", "city"],
  ["france", "japan"],
  ["dog", "cat", "animal"],
  ["dog", "animal"],
  ["cat", "animal"],
].map((s) => s.map(idx));

export interface W2VOptions {
  seed?: number;
  epochs?: number;
  lr?: number;
}

export function runWord2Vec(opts: W2VOptions = {}): Trajectory<EmbeddingState> {
  const { seed = (Date.now() & 0xffff) >>> 0, epochs = 220, lr = 0.1 } = opts;
  const rng = mulberry32(seed);
  const V = WORDS.length;
  const rnd = () => (rng() * 2 - 1) * 0.4;
  const vIn = Array.from({ length: V }, () => [rnd(), rnd()]); // 中心词向量（用于可视化）
  const uOut = Array.from({ length: V }, () => [rnd(), rnd()]); // 上下文词向量

  // 训练对 (center, context)
  const pairs: [number, number][] = [];
  for (const s of SENTENCES) for (const c of s) for (const o of s) if (c !== o) pairs.push([c, o]);

  const frames: Frame<EmbeddingState>[] = [];
  const snapshot = (iter: number, loss: number) =>
    frames.push({
      iter,
      state: { words: WORDS, positions: vIn.map(([x, y]) => ({ x, y })), groups: GROUPS },
      metrics: { loss },
    });
  snapshot(0, 0);

  for (let e = 1; e <= epochs; e++) {
    let loss = 0;
    for (const [c, o] of pairs) {
      // softmax over vocab: p(k|c) ∝ exp(vIn_c · uOut_k)
      const scores = uOut.map((u) => u[0] * vIn[c][0] + u[1] * vIn[c][1]);
      const mx = Math.max(...scores);
      const exp = scores.map((s) => Math.exp(s - mx));
      const Z = exp.reduce((a, b) => a + b, 0);
      const p = exp.map((x) => x / Z);
      loss += -Math.log(p[o] + 1e-9);

      // 梯度
      const dvc = [0, 0];
      for (let k = 0; k < V; k++) {
        const g = p[k] - (k === o ? 1 : 0);
        uOut[k][0] -= lr * g * vIn[c][0];
        uOut[k][1] -= lr * g * vIn[c][1];
        dvc[0] += g * uOut[k][0];
        dvc[1] += g * uOut[k][1];
      }
      vIn[c][0] -= lr * dvc[0];
      vIn[c][1] -= lr * dvc[1];
    }
    if (e % 5 === 0) snapshot(e, loss / pairs.length);
  }

  return {
    meta: {
      id: "word2vec",
      title: "Word2Vec 词向量",
      family: "embedding",
      algorithm: "Word2Vec (skip-gram)",
      description: "在小语料上学词向量。经常一起出现的词越靠越近，同语义的词聚成一团。",
      tutorial: {
        problem: "怎么让机器理解“词的含义”？Word2Vec：用一个向量表示每个词，让相关的词向量彼此靠近。",
        intuition:
          "核心假设：“看一个词交什么朋友，就知道它什么意思”（分布式语义）。训练时让每个词去预测它的上下文词，于是经常共现的词（king/queen、paris/france）向量被拉到一起。学出来的向量空间还有惊人的代数性质：king − man + woman ≈ queen。这是 2013 年让 NLP 起飞的关键。",
        watch: [
          "每个点是一个词，颜色=语义组（王室/地理/动物）",
          "初始随机散布，随训练同组的词逐渐聚到一起",
          "右侧 loss 下降 = 词越来越能预测自己的上下文",
        ],
        concepts: [
          { term: "词向量 / 嵌入", explain: "用一个稠密向量表示词，相近含义→相近向量" },
          { term: "skip-gram", explain: "用中心词预测周围上下文词，从共现中学语义" },
          { term: "分布式语义", explain: "词义由它出现的上下文决定（近朱者赤）" },
        ],
        tryThis: "拖时间轴看三组词从混杂到聚成三团；点“重新生成数据”换初始化重学。",
      },
      hyperparams: { vocab: V, dim: 2, epochs, lr },
    },
    frames,
  };
}
