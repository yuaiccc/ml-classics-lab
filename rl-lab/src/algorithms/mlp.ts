// 从零手写的多层感知机（MLP）+ 反向传播，无任何深度学习库。
// 在同心圆数据（线性不可分）上训练，每个 epoch 记录决策边界 —— 展示神经网络如何学出非线性边界。
import { Trajectory, BoundaryState, Frame } from "@/player/types";
import { makeCircles } from "./datasets";
import { computeBounds, makeGrid, sigmoid } from "./grid";
import { mulberry32 } from "./rng";

export interface MLPOptions {
  seed?: number;
  hidden?: number;
  lr?: number;
  epochs?: number;
}

export function runMLP(opts: MLPOptions = {}): Trajectory<BoundaryState> {
  const { seed = (Date.now() & 0xffff) >>> 0, hidden = 12, lr = 0.5, epochs = 120 } = opts;
  const points = makeCircles(seed);
  const bounds = computeBounds(points);
  const rng = mulberry32(seed);

  // 输入标准化（否则 tanh 容易饱和）
  const mx = points.reduce((s, p) => s + p.x, 0) / points.length;
  const my = points.reduce((s, p) => s + p.y, 0) / points.length;
  const sx = Math.sqrt(points.reduce((s, p) => s + (p.x - mx) ** 2, 0) / points.length) || 1;
  const sy = Math.sqrt(points.reduce((s, p) => s + (p.y - my) ** 2, 0) / points.length) || 1;
  const norm = (x: number, y: number): [number, number] => [(x - mx) / sx, (y - my) / sy];

  // 参数：隐藏层 W1[h][2], b1[h]（tanh）；输出层 W2[h], b2（sigmoid）
  const H = hidden;
  const rand = () => (rng() * 2 - 1) * 0.8;
  const W1 = Array.from({ length: H }, () => [rand(), rand()]);
  const b1 = Array.from({ length: H }, () => 0);
  const W2 = Array.from({ length: H }, () => rand());
  let b2 = 0;

  // 前向：返回 (a1 隐藏激活, prob 输出概率)
  const forward = (xn: number, yn: number): { a1: number[]; prob: number } => {
    const a1 = new Array(H);
    for (let j = 0; j < H; j++) {
      a1[j] = Math.tanh(W1[j][0] * xn + W1[j][1] * yn + b1[j]);
    }
    let z2 = b2;
    for (let j = 0; j < H; j++) z2 += W2[j] * a1[j];
    return { a1, prob: sigmoid(z2) };
  };

  const frames: Frame<BoundaryState>[] = [];

  for (let e = 0; e <= epochs; e++) {
    // —— 记录这一帧的决策边界 + 指标 ——
    const grid = makeGrid(bounds, 44, 34, (x, y) => {
      const [xn, yn] = norm(x, y);
      return forward(xn, yn).prob;
    });
    let loss = 0;
    let correct = 0;
    for (const p of points) {
      const [xn, yn] = norm(p.x, p.y);
      const { prob } = forward(xn, yn);
      loss += -(p.label * Math.log(prob + 1e-9) + (1 - p.label) * Math.log(1 - prob + 1e-9));
      if ((prob >= 0.5 ? 1 : 0) === p.label) correct++;
    }
    frames.push({
      iter: e,
      state: { points, grid },
      metrics: { loss: loss / points.length, accuracy: correct / points.length },
    });
    if (e === epochs) break;

    // —— 反向传播（全批量梯度下降）——
    const gW1 = Array.from({ length: H }, () => [0, 0]);
    const gb1 = new Array(H).fill(0);
    const gW2 = new Array(H).fill(0);
    let gb2 = 0;
    for (const p of points) {
      const [xn, yn] = norm(p.x, p.y);
      const { a1, prob } = forward(xn, yn);
      const dz2 = prob - p.label; // BCE + sigmoid 的简化梯度
      for (let j = 0; j < H; j++) {
        gW2[j] += dz2 * a1[j];
        const dz1 = dz2 * W2[j] * (1 - a1[j] * a1[j]); // tanh'
        gW1[j][0] += dz1 * xn;
        gW1[j][1] += dz1 * yn;
        gb1[j] += dz1;
      }
      gb2 += dz2;
    }
    const n = points.length;
    for (let j = 0; j < H; j++) {
      W2[j] -= (lr * gW2[j]) / n;
      W1[j][0] -= (lr * gW1[j][0]) / n;
      W1[j][1] -= (lr * gW1[j][1]) / n;
      b1[j] -= (lr * gb1[j]) / n;
    }
    b2 -= (lr * gb2) / n;
  }

  return {
    meta: {
      id: "mlp",
      title: "多层感知机 MLP（手写反向传播）",
      family: "scatter-boundary",
      algorithm: "MLP",
      description: "同心圆数据。一个 2→隐藏层(tanh)→1(sigmoid) 的小神经网络，反向传播逐 epoch 学出弯曲边界。",
      tutorial: {
        problem: "逻辑回归只能画直线，分不开同心圆。神经网络靠隐藏层学出弯曲的决策边界。",
        intuition:
          "隐藏层每个神经元先用 tanh 把空间“掰弯”一下，输出层再把这些弯曲特征组合起来——多个非线性叠加，就能拟合环形这种复杂边界。反向传播负责把误差从输出一层层传回去，告诉每个权重该怎么调。",
        watch: [
          "边界一开始是模糊的直线/曲线，随 epoch 逐渐包出中间的圆",
          "Loss 下降、准确率上升，最终能把内圈外环基本分开",
          "对比“逻辑回归”：那条永远是直线，这里能弯——这就是隐藏层的威力",
        ],
        concepts: [
          { term: "隐藏层", explain: "中间一层神经元，用非线性激活(tanh)制造弯曲特征" },
          { term: "反向传播", explain: "用链式法则把输出误差逐层传回，算出每个权重的梯度" },
          { term: "激活函数", explain: "tanh/sigmoid 这类非线性函数，没有它多层等于一层" },
        ],
        tryThis: "拖时间轴看边界从直到弯的全过程；再点“重新生成数据”换一组圆，看它每次都能学出环形。",
      },
      hyperparams: { hidden: H, lr, epochs, points: points.length },
    },
    frames,
  };
}
