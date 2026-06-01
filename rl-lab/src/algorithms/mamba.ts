// Mamba 的核心：选择性状态空间扫描（selective scan）—— 浏览器端纯手写，无库。
// 任务：序列里信号 token（A–D）散落在空白（·）中，从左到右扫一遍，
// 「选择门 Δ」根据当前输入决定是“写入记忆”还是“忽略”，于是记忆始终 latch 住最近的信号。
// 这就是 Mamba 区别于 Transformer 的地方：线性时间的选择性递归，而非 O(L²) 注意力矩阵。
import { Trajectory, SSMState, Frame } from "@/player/types";
import { mulberry32 } from "./rng";

const CHANNELS = 4; // 信号种类 A,B,C,D

export interface MambaOptions {
  seed?: number;
  length?: number;
  signals?: number;
}

export function runSelectiveSSM(opts: MambaOptions = {}): Trajectory<SSMState> {
  const { seed = (Date.now() & 0xffff) >>> 0, length = 14, signals = 4 } = opts;
  const rng = mulberry32(seed);

  // 生成序列：signals 个非零信号散落在空白中
  const tokens = new Array(length).fill(0);
  const used: number[] = [];
  while (used.length < Math.min(signals, length)) {
    const p = Math.floor(rng() * length);
    if (!used.includes(p)) used.push(p);
  }
  for (const p of used) tokens[p] = 1 + Math.floor(rng() * CHANNELS); // 1..4

  const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

  // 一次性扫描整条序列（O(L) 递归）
  const gates: number[] = [];
  const memory: number[][] = [];
  const output: number[] = [];
  let m = new Array(CHANNELS).fill(0);

  for (let t = 0; t < length; t++) {
    const tok = tokens[t];
    // 选择门 Δ：随输入变化（信号→接近1=写入；空白→接近0=保持）。这就是“selective”。
    const g = sigmoid(tok > 0 ? 4 : -4);
    const onehot = new Array(CHANNELS).fill(0);
    if (tok > 0) onehot[tok - 1] = 1;
    // 状态更新： h_t = (1-Δ)·h_{t-1} + Δ·x_t  —— 门高则覆写、门低则保持
    m = m.map((v, i) => (1 - g) * v + g * onehot[i]);

    gates.push(g);
    memory.push([...m]);
    let best = 0;
    let bv = -1;
    m.forEach((v, i) => {
      if (v > bv) {
        bv = v;
        best = i;
      }
    });
    output.push(bv > 0.05 ? best + 1 : 0);
  }

  const frames: Frame<SSMState>[] = [];
  for (let t = 0; t < length; t++) {
    frames.push({
      iter: t,
      state: { tokens, pos: t, gates, memory, output },
      metrics: { gate: gates[t] },
    });
  }

  return {
    meta: {
      id: "mamba-ssm",
      title: "Mamba · 选择性状态空间",
      family: "ssm",
      algorithm: "Selective SSM (Mamba)",
      description:
        "序列里信号散落在空白中。选择性扫描从左到右走一遍，input-dependent 的门控决定记什么、忘什么。",
      tutorial: {
        problem:
          "如何用线性时间（不靠 O(L²) 注意力）在长序列里只记住重要信息、忽略噪声？这是 Mamba 想解决的。",
        intuition:
          "想象从左到右读序列，手里有一块“记忆”。每读一个 token，一个“选择门 Δ”根据当前输入决定：要不要把它写进记忆。信号 token → 门开（覆写记忆）；空白噪声 → 门关（记忆原样保持）。所以记忆里始终是“最近看到的信号”。这套选择性递归就是 Mamba 的发动机，和注意力目标相同（跨位置传信息），但只需扫一遍、线性时间。",
        watch: [
          "上排是输入序列：字母=信号，· =空白噪声；高亮列是当前扫描位置",
          "中间“选择门 Δ”：信号位置门开（高柱），空白位置门关（矮柱）",
          "下方记忆热图：信号一来就被写入并一直保持，直到下一个信号覆写——空白完全不影响它",
          "对比注意力：那是 L×L 的“谁看谁”矩阵；这里只有一遍从左到右的扫描",
        ],
        concepts: [
          { term: "选择门 Δ", explain: "随输入变化的门控，决定当前 token 写入记忆的强度（selective 的来源）" },
          { term: "状态/记忆 h", explain: "一个随序列递推更新的隐藏状态，浓缩了“到目前为止的重要信息”" },
          { term: "线性时间 O(L)", explain: "只扫一遍序列，不像注意力要算 L×L，所以适合超长序列" },
        ],
        tryThis: "拖时间轴看记忆如何被信号“点亮”、在空白处保持不变；点“重新生成数据”换一组信号分布。",
      },
      hyperparams: { length, signals, channels: CHANNELS },
    },
    frames,
  };
}
