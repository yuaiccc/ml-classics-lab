// Iris 鸢尾花 · Softmax 多分类（多项逻辑回归）—— 真实数据集，浏览器手写。
// 用花瓣长/宽两个特征（最能分开三类），梯度下降学出三类决策边界。
import { Trajectory, MultiBoundaryState, Frame } from "@/player/types";
import { IRIS, IRIS_CLASSES, IRIS_FEATURES } from "@/data/iris";
import { computeBounds, makeGrid } from "./grid";

const FX = 2; // 花瓣长
const FY = 3; // 花瓣宽
const K = 3; // 三类

export interface IrisSoftmaxOptions {
  lr?: number;
  epochs?: number;
}

export function runIrisSoftmax(opts: IrisSoftmaxOptions = {}): Trajectory<MultiBoundaryState> {
  const { lr = 0.5, epochs = 80 } = opts;

  const pts = IRIS.map((d) => ({ x: d.x[FX], y: d.x[FY], label: d.y }));
  const bounds = computeBounds(pts);

  // 标准化两个特征
  const mean = [avg(pts.map((p) => p.x)), avg(pts.map((p) => p.y))];
  const std = [stdv(pts.map((p) => p.x), mean[0]), stdv(pts.map((p) => p.y), mean[1])];
  const norm = (x: number, y: number): [number, number] => [(x - mean[0]) / std[0], (y - mean[1]) / std[1]];

  // 参数：每类一个线性判别 W[k]·x + b[k]
  const W = Array.from({ length: K }, () => [0, 0]);
  const b = new Array(K).fill(0);

  const probs = (xn: number, yn: number) => {
    const logits = W.map((w, k) => w[0] * xn + w[1] * yn + b[k]);
    const mx = Math.max(...logits);
    const exp = logits.map((l) => Math.exp(l - mx));
    const Z = exp.reduce((a, c) => a + c, 0);
    return exp.map((e) => e / Z);
  };
  const predict = (x: number, y: number) => {
    const [xn, yn] = norm(x, y);
    const p = probs(xn, yn);
    return p.indexOf(Math.max(...p));
  };

  const frames: Frame<MultiBoundaryState>[] = [];
  for (let e = 0; e <= epochs; e++) {
    const grid = makeGrid(bounds, 48, 38, (x, y) => predict(x, y));
    let loss = 0;
    let correct = 0;
    for (const pt of pts) {
      const [xn, yn] = norm(pt.x, pt.y);
      const p = probs(xn, yn);
      loss += -Math.log(p[pt.label] + 1e-9);
      if (p.indexOf(Math.max(...p)) === pt.label) correct++;
    }
    frames.push({
      iter: e,
      state: { points: pts, grid, classNames: IRIS_CLASSES, xName: IRIS_FEATURES[FX], yName: IRIS_FEATURES[FY] },
      metrics: { loss: loss / pts.length, accuracy: correct / pts.length },
    });
    if (e === epochs) break;

    // 梯度
    const gW = W.map(() => [0, 0]);
    const gb = new Array(K).fill(0);
    for (const pt of pts) {
      const [xn, yn] = norm(pt.x, pt.y);
      const p = probs(xn, yn);
      for (let k = 0; k < K; k++) {
        const g = p[k] - (k === pt.label ? 1 : 0);
        gW[k][0] += g * xn;
        gW[k][1] += g * yn;
        gb[k] += g;
      }
    }
    const n = pts.length;
    for (let k = 0; k < K; k++) {
      W[k][0] -= (lr * gW[k][0]) / n;
      W[k][1] -= (lr * gW[k][1]) / n;
      b[k] -= (lr * gb[k]) / n;
    }
  }

  return {
    meta: {
      id: "iris-softmax",
      title: "Iris · Softmax 多分类",
      family: "multiclass",
      algorithm: "Softmax Regression",
      description: "真实数据集（150 朵鸢尾花）。用花瓣长/宽两个特征，softmax 学出三类决策边界。",
      tutorial: {
        problem: "经典真实数据：根据花瓣尺寸把鸢尾花分成 setosa / versicolor / virginica 三类。",
        intuition:
          "Softmax 回归是逻辑回归的多类版本：给每个类别一条线性判别，再用 softmax 把三个分数变成三个概率，取最大的当预测。训练让每个样本的正确类别概率尽量大。setosa 离得远很好分，另外两类有重叠、边界更难。",
        watch: [
          "三种颜色的点是真实的三类鸢尾花（花瓣长 vs 花瓣宽）",
          "背景三色区是当前决策边界，随训练逐渐把三类分开",
          "setosa（左下）很快被分出；versicolor / virginica 交界处会残留少量错分",
        ],
        concepts: [
          { term: "Softmax", explain: "把多个分数归一成和为 1 的概率分布，多分类版 sigmoid" },
          { term: "决策边界", explain: "相邻两类概率相等处，三类两两之间各有一条" },
          { term: "真实数据集", explain: "Iris 是 1936 年 Fisher 提出的经典基准，ML 界 Hello World" },
        ],
        tryThis: "拖时间轴看三色边界从无到成形；注意 versicolor / virginica 交界永远分不干净（数据本身重叠）。",
      },
      hyperparams: { features: "花瓣长/宽", lr, epochs, samples: IRIS.length },
    },
    frames,
  };
}

const avg = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;
const stdv = (a: number[], m: number) => Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length) || 1;
