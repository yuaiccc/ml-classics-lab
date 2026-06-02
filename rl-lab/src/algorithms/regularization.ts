// 正则化（L2 / 岭回归）—— 固定高阶多项式，正则强度 λ 从小到大。
// λ 小=过拟合（曲线乱扭），λ 大=欠拟合（曲线变平）；中间有个最佳点。
import { Trajectory, CurveFitState, Frame } from "@/player/types";
import { mulberry32, gaussian } from "./rng";
import { polyfit, polyval } from "./linalg";

const truth = (x: number) => Math.sin(2.6 * x);

export interface RegOptions {
  seed?: number;
  degree?: number;
  steps?: number;
}

export function runRegularization(opts: RegOptions = {}): Trajectory<CurveFitState> {
  const { seed = (Date.now() & 0xffff) >>> 0, degree = 12, steps = 26 } = opts;
  const rng = mulberry32(seed);

  const trainX = Array.from({ length: 14 }, () => (rng() * 2 - 1) * 1.2).sort((a, b) => a - b);
  const trainY = trainX.map((x) => truth(x) + gaussian(rng, 0, 0.18));
  const testX = Array.from({ length: 120 }, (_, i) => -1.2 + (2.4 * i) / 119);
  const testY = testX.map((x) => truth(x) + gaussian(rng, 0, 0.18));
  const truthCurve = testX.map((x) => ({ x, y: truth(x) }));
  const mse = (xs: number[], ys: number[], w: number[]) =>
    xs.reduce((s, x, i) => s + (polyval(w, x) - ys[i]) ** 2, 0) / xs.length;

  const frames: Frame<CurveFitState>[] = [];
  for (let i = 0; i < steps; i++) {
    // λ 从 1e-6 对数增长到 ~30
    const lambda = Math.pow(10, -6 + (7.5 * i) / (steps - 1));
    const w = polyfit(trainX, trainY, degree, lambda);
    const fit = testX.map((x) => ({ x, y: Math.max(-3, Math.min(3, polyval(w, x))) }));
    frames.push({
      iter: i,
      state: {
        train: trainX.map((x, k) => ({ x, y: trainY[k] })),
        test: testX.map((x, k) => ({ x, y: testY[k] })),
        fit,
        truth: truthCurve,
        caption: `λ = ${lambda.toExponential(1)}`,
      },
      metrics: { trainError: mse(trainX, trainY, w), testError: mse(testX, testY, w), logLambda: Math.log10(lambda) },
    });
  }

  return {
    meta: {
      id: "regularization",
      title: "正则化 (L2 / 岭回归)",
      family: "curvefit",
      algorithm: "Ridge Regression",
      description: "固定 12 阶多项式（很容易过拟合），增大正则强度 λ，看它怎么把曲线“拉平”、防过拟合。",
      tutorial: {
        problem: "模型太复杂会过拟合，又不想降低复杂度怎么办？正则化：给“扭动”加惩罚。",
        intuition:
          "在损失里加一项 λ‖w‖²（L2）：权重越大惩罚越重，于是模型倾向于用更小、更平滑的权重。λ 小时几乎没约束、曲线乱扭（过拟合）；λ 增大曲线被逐渐“拉平”；λ 太大则过度约束、退化成直线（欠拟合）。L1（Lasso）类似但会把部分权重压成 0（稀疏）。",
        watch: [
          "λ 从极小增大，绿色拟合曲线从剧烈扭动逐渐变平滑",
          "右侧测试误差先降后升：存在一个最佳 λ",
          "λ 过大时曲线几乎成直线 = 欠拟合",
        ],
        concepts: [
          { term: "L2 / 岭回归", explain: "惩罚权重平方和，让权重整体变小、模型更平滑" },
          { term: "L1 / Lasso", explain: "惩罚权重绝对值，会把部分权重压成 0，做特征选择" },
          { term: "正则强度 λ", explain: "惩罚的力度；需要在欠拟合和过拟合之间调到最佳" },
        ],
        tryThis: "拖时间轴找测试误差最低的 λ；对比「过拟合」实验——一个调复杂度、一个调正则，殊途同归。",
      },
      hyperparams: { degree, steps, lambdaRange: "1e-6 → ~30" },
    },
    frames,
  };
}
