// 过拟合与泛化 —— 多项式拟合，阶数从低到高。
// 阶数太低=欠拟合，太高=过拟合：训练误差一路降，但测试误差先降后升（经典 U 形）。
import { Trajectory, CurveFitState, Frame } from "@/player/types";
import { mulberry32, gaussian } from "./rng";
import { polyfit, polyval } from "./linalg";

const truth = (x: number) => Math.sin(2.6 * x); // 真实函数

export interface OverfitOptions {
  seed?: number;
  nTrain?: number;
  maxDeg?: number;
  noise?: number;
}

export function runOverfitting(opts: OverfitOptions = {}): Trajectory<CurveFitState> {
  const { seed = (Date.now() & 0xffff) >>> 0, nTrain = 14, maxDeg = 13, noise = 0.18 } = opts;
  const rng = mulberry32(seed);

  const X = (n: number) => Array.from({ length: n }, () => (rng() * 2 - 1) * 1.2); // x ∈ [-1.2, 1.2]
  const trainX = X(nTrain).sort((a, b) => a - b);
  const trainY = trainX.map((x) => truth(x) + gaussian(rng, 0, noise));
  const testX = Array.from({ length: 120 }, (_, i) => -1.2 + (2.4 * i) / 119);
  const testY = testX.map((x) => truth(x) + gaussian(rng, 0, noise));

  const truthCurve = testX.map((x) => ({ x, y: truth(x) }));
  const mse = (xs: number[], ys: number[], w: number[]) =>
    xs.reduce((s, x, i) => s + (polyval(w, x) - ys[i]) ** 2, 0) / xs.length;

  const frames: Frame<CurveFitState>[] = [];
  for (let deg = 1; deg <= maxDeg; deg++) {
    const w = polyfit(trainX, trainY, deg, 1e-7); // 极小正则仅为数值稳定
    const fit = testX.map((x) => ({ x, y: Math.max(-3, Math.min(3, polyval(w, x))) }));
    frames.push({
      iter: deg,
      state: {
        train: trainX.map((x, i) => ({ x, y: trainY[i] })),
        test: testX.map((x, i) => ({ x, y: testY[i] })),
        fit,
        truth: truthCurve,
        caption: `多项式阶数 = ${deg}`,
      },
      metrics: { trainError: mse(trainX, trainY, w), testError: mse(testX, testY, w) },
    });
  }

  return {
    meta: {
      id: "overfitting",
      title: "过拟合与泛化",
      family: "curvefit",
      algorithm: "Polynomial Fit",
      description: "用多项式拟合带噪声的曲线。阶数越高越能贴合训练点，但太高就过拟合，测试误差反升。",
      tutorial: {
        problem: "模型越复杂越好吗？这个实验展示「过拟合」——在训练集上完美，到新数据上反而更差。",
        intuition:
          "用阶数越来越高的多项式去拟合同一批点。低阶（直线）太简单，拟合不上真实曲线（欠拟合）；阶数升高慢慢贴合；阶数过高时，曲线为了穿过每个带噪训练点而剧烈扭动，把噪声也学了进去（过拟合），对新数据预测很差。",
        watch: [
          "灰色虚线=真实函数，青点=训练点，绿线=当前阶数的拟合",
          "阶数升高，绿线从直变弯、越来越贴训练点，最后疯狂扭动",
          "右侧两条曲线：训练误差一路降，但测试误差先降后升 = 经典 U 形",
        ],
        concepts: [
          { term: "欠拟合", explain: "模型太简单，连训练数据的规律都没学到" },
          { term: "过拟合", explain: "模型太复杂，把训练数据的噪声也背了下来，泛化差" },
          { term: "泛化", explain: "在没见过的新数据上的表现，才是真正关心的" },
        ],
        tryThis: "拖时间轴找测试误差最低的那个阶数（最佳复杂度）；继续加阶看曲线怎么疯狂扭动。",
      },
      hyperparams: { nTrain, maxDeg, noise },
    },
    frames,
  };
}
