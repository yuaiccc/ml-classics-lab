// 香草 RNN + 时序反向传播(BPTT) —— 浏览器手写。
// 历史桥梁：MLP→[RNN]→Transformer。任务=奇偶校验（数序列里有几个 1，奇/偶）——
// 前馈网络做不到（要跨时间记忆），RNN 靠隐藏状态把信息一路带下去。
import { Trajectory, RnnState, Frame } from "@/player/types";
import { mulberry32 } from "./rng";

const L = 6; // 序列长度
const H = 10; // 隐藏维度

export interface RnnOptions {
  seed?: number;
  epochs?: number;
  lr?: number;
}

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

export function runRNN(opts: RnnOptions = {}): Trajectory<RnnState> {
  const { seed = (Date.now() & 0xffff) >>> 0, epochs = 140, lr = 0.15 } = opts;
  const rng = mulberry32(seed);
  const rnd = () => (rng() * 2 - 1) * 0.5;

  // 参数
  const Wxh = Array.from({ length: H }, () => [rnd()]);
  const Whh = Array.from({ length: H }, () => Array.from({ length: H }, rnd));
  const bh = new Array(H).fill(0);
  const Why = Array.from({ length: H }, rnd); // H -> 1
  let by = 0;

  // 数据：随机比特序列，标签=奇偶
  const data = Array.from({ length: 200 }, () => {
    const x = Array.from({ length: L }, () => (rng() < 0.5 ? 0 : 1));
    return { x, y: x.reduce((a, b) => a + b, 0) % 2 };
  });

  const forward = (x: number[]) => {
    const hs: number[][] = [new Array(H).fill(0)];
    for (let t = 0; t < L; t++) {
      const h = new Array(H);
      for (let j = 0; j < H; j++) {
        let a = bh[j] + Wxh[j][0] * x[t];
        for (let k = 0; k < H; k++) a += Whh[j][k] * hs[t][k];
        h[j] = Math.tanh(a);
      }
      hs.push(h);
    }
    let logit = by;
    for (let j = 0; j < H; j++) logit += Why[j] * hs[L][j];
    return { hs, p: sigmoid(logit) };
  };

  const fixed = data[0]; // 固定样本用于可视化

  const snapshot = (iter: number) => {
    let loss = 0;
    let correct = 0;
    for (const d of data) {
      const { p } = forward(d.x);
      loss += -(d.y * Math.log(p + 1e-9) + (1 - d.y) * Math.log(1 - p + 1e-9));
      if ((p >= 0.5 ? 1 : 0) === d.y) correct++;
    }
    const { hs, p } = forward(fixed.x);
    frames.push({
      iter,
      state: { inputs: fixed.x, hidden: hs.slice(1), target: fixed.y, pred: p >= 0.5 ? 1 : 0 },
      metrics: { accuracy: correct / data.length, loss: loss / data.length },
    });
  };

  const frames: Frame<RnnState>[] = [];
  snapshot(0);

  for (let e = 1; e <= epochs; e++) {
    // 累积梯度（全批量 BPTT）
    const gWxh = Wxh.map((r) => r.map(() => 0));
    const gWhh = Whh.map((r) => r.map(() => 0));
    const gbh = new Array(H).fill(0);
    const gWhy = new Array(H).fill(0);
    let gby = 0;

    for (const d of data) {
      const { hs, p } = forward(d.x);
      const dlogit = p - d.y;
      gby += dlogit;
      let dh = new Array(H).fill(0);
      for (let j = 0; j < H; j++) {
        gWhy[j] += dlogit * hs[L][j];
        dh[j] = Why[j] * dlogit;
      }
      for (let t = L - 1; t >= 0; t--) {
        const hCur = hs[t + 1];
        const hPrev = hs[t];
        const dhPrev = new Array(H).fill(0);
        for (let j = 0; j < H; j++) {
          const da = dh[j] * (1 - hCur[j] * hCur[j]);
          gbh[j] += da;
          gWxh[j][0] += da * d.x[t];
          for (let k = 0; k < H; k++) {
            gWhh[j][k] += da * hPrev[k];
            dhPrev[k] += Whh[j][k] * da;
          }
        }
        dh = dhPrev;
      }
    }

    const n = data.length;
    for (let j = 0; j < H; j++) {
      Wxh[j][0] -= (lr * gWxh[j][0]) / n;
      bh[j] -= (lr * gbh[j]) / n;
      Why[j] -= (lr * gWhy[j]) / n;
      for (let k = 0; k < H; k++) Whh[j][k] -= (lr * gWhh[j][k]) / n;
    }
    by -= (lr * gby) / n;

    if (e % 4 === 0) snapshot(e);
  }

  return {
    meta: {
      id: "rnn",
      title: "RNN · 序列记忆",
      family: "rnn",
      algorithm: "Vanilla RNN (BPTT)",
      description: "任务=奇偶校验。前馈网络做不到（需跨时间记忆），RNN 用隐藏状态把信息一路带下去。",
      tutorial: {
        problem: "判断一串比特里 1 的个数是奇是偶——必须“记住”前面看过什么。普通前馈网络没有记忆，做不到。",
        intuition:
          "RNN 有一个隐藏状态 h，像一块随时间更新的便签：每读一个新输入，就把它和上一刻的 h 结合、更新成新的 h。信息就这样沿时间一路传递。训练靠 BPTT——把误差沿时间倒着传回每一步。这是 Transformer 之前处理序列的主力。",
        watch: [
          "上排是输入比特序列；下方热图是隐藏状态随时间(从左到右)的变化",
          "训练初期隐藏状态杂乱、预测乱猜；随 epoch 隐藏状态学出规律",
          "右侧准确率升到接近 1 = RNN 学会了跨时间数奇偶",
        ],
        concepts: [
          { term: "隐藏状态 h", explain: "随时间更新的记忆，浓缩了到目前为止的序列信息" },
          { term: "循环连接", explain: "上一刻的 h 反馈进当前计算，形成记忆" },
          { term: "BPTT", explain: "时序反向传播：把误差沿时间一步步倒传回去更新参数" },
        ],
        tryThis: "拖时间轴看准确率从 0.5（瞎猜）爬到接近 1；对比 Self-Attention——都处理序列，但机制不同。",
      },
      hyperparams: { seqLen: L, hidden: H, epochs, lr },
    },
    frames,
  };
}
