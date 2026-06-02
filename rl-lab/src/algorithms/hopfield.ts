// Hopfield 网络（1982）—— 联想记忆。存几张图案，给一张残缺/带噪的，
// 网络靠能量下降把它“回忆”成最接近的完整图案。浏览器手写。
import { Trajectory, HopfieldState, Frame } from "@/player/types";
import { mulberry32 } from "./rng";

const N = 9; // 图案边长
const SIZE = N * N;

// 三张存储的图案（# = +1，. = -1）
const PATTERNS = [
  ["....#....", "....#....", "....#....", "....#....", "#########", "....#....", "....#....", "....#....", "....#...."],
  ["#########", "#.......#", "#.......#", "#.......#", "#.......#", "#.......#", "#.......#", "#.......#", "#########"],
  ["#.......#", ".#.....#.", "..#...#..", "...#.#...", "....#....", "...#.#...", "..#...#..", ".#.....#.", "#.......#"],
].map((rows) => rows.join("").split("").map((c) => (c === "#" ? 1 : -1)));

export interface HopfieldOptions {
  seed?: number;
  noise?: number; // 翻转比例
}

export function runHopfield(opts: HopfieldOptions = {}): Trajectory<HopfieldState> {
  const { seed = (Date.now() & 0xffff) >>> 0, noise = 0.28 } = opts;
  const rng = mulberry32(seed);

  // Hebbian 权重：W = Σ_p p pᵀ（对角置零）
  const W: number[][] = Array.from({ length: SIZE }, () => new Array(SIZE).fill(0));
  for (const p of PATTERNS) {
    for (let i = 0; i < SIZE; i++)
      for (let j = 0; j < SIZE; j++) if (i !== j) W[i][j] += (p[i] * p[j]) / SIZE;
  }

  // 随机选一张目标图案，加噪作为初始状态
  const target = PATTERNS[Math.floor(rng() * PATTERNS.length)];
  let s = target.map((v) => (rng() < noise ? -v : v));

  const energy = (st: number[]) => {
    let e = 0;
    for (let i = 0; i < SIZE; i++) for (let j = 0; j < SIZE; j++) e -= 0.5 * W[i][j] * st[i] * st[j];
    return e;
  };
  const overlap = (st: number[]) => st.reduce((acc, v, i) => acc + (v === target[i] ? 1 : 0), 0) / SIZE;

  const frames: Frame<HopfieldState>[] = [];
  const snap = (iter: number) =>
    frames.push({
      iter,
      state: { size: N, cells: [...s], target },
      metrics: { overlap: overlap(s), energy: energy(s) },
    });
  snap(0);

  // 同步更新，直到稳定
  for (let it = 1; it <= 8; it++) {
    const next = s.map((_, i) => {
      let sum = 0;
      for (let j = 0; j < SIZE; j++) sum += W[i][j] * s[j];
      return sum >= 0 ? 1 : -1;
    });
    const changed = next.some((v, i) => v !== s[i]);
    s = next;
    snap(it);
    if (!changed) break;
  }

  return {
    meta: {
      id: "hopfield",
      title: "Hopfield 联想记忆",
      family: "hopfield",
      algorithm: "Hopfield Network",
      description: "存了 3 张图案。给一张残缺/带噪的，网络靠能量下降把它“回忆”成完整图案。",
      tutorial: {
        problem: "人看到半张脸也能认出是谁。机器能不能从残缺/带噪的输入“回忆”出完整记忆？",
        intuition:
          "Hopfield 网络把要记的图案编码进神经元之间的连接权重（Hebbian：一起激活的连在一起）。给一个带噪的初始状态，网络反复让每个神经元顺从邻居的“多数意见”更新，整个系统的“能量”单调下降，最终落到最近的一个记忆图案上——就像小球滚进最近的山谷。",
        watch: [
          "第 0 帧是带噪声的残缺图案（约 28% 像素翻转）",
          "每次更新都把它往最近的记忆图案拉，几步内完全恢复",
          "右侧 overlap（与目标的吻合度）升到 1 = 完美回忆",
        ],
        concepts: [
          { term: "联想记忆", explain: "用内容的一部分检索出完整内容，而非按地址" },
          { term: "Hebbian 学习", explain: "“一起激活的神经元连接变强”，把图案存进权重" },
          { term: "能量函数", explain: "系统总能量随更新单调下降，收敛到记忆（吸引子）" },
        ],
        tryThis: "点“重新生成数据”换不同的噪声/目标图案，看它每次都能回忆出最接近的那张。",
      },
      hyperparams: { patterns: PATTERNS.length, size: `${N}×${N}`, noise },
    },
    frames,
  };
}
