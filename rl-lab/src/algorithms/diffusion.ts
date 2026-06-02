// 扩散模型（DDPM 简化版）—— 浏览器手写。Stable Diffusion 的底层思想：
// 训练一个“去噪器”，然后从纯噪声出发一步步去噪，生成目标分布（two-moons 双月形）。
import { Trajectory, ClusterState, Frame, Point2D } from "@/player/types";
import { mulberry32, gaussian } from "./rng";
import { MLP, makeMLP, forward, zeroGrads, backwardAccum, applyGrads } from "./nn";

export interface DiffusionOptions {
  seed?: number;
  steps?: number; // 扩散时间步 T
  trainIters?: number;
}

function makeMoons(rng: () => number, n = 140): Point2D[] {
  const pts: Point2D[] = [];
  for (let i = 0; i < n; i++) {
    const t = (Math.PI * i) / (n - 1);
    if (i % 2 === 0) pts.push({ x: Math.cos(t), y: Math.sin(t) });
    else pts.push({ x: 1 - Math.cos(t), y: 0.5 - Math.sin(t) });
  }
  // 居中 + 缩放 + 加噪
  const mx = pts.reduce((s, p) => s + p.x, 0) / pts.length;
  const my = pts.reduce((s, p) => s + p.y, 0) / pts.length;
  return pts.map((p) => ({ x: (p.x - mx) * 3 + gaussian(rng, 0, 0.12), y: (p.y - my) * 3 + gaussian(rng, 0, 0.12) }));
}

export function runDiffusion(opts: DiffusionOptions = {}): Trajectory<ClusterState> {
  const { seed = (Date.now() & 0xffff) >>> 0, steps = 16, trainIters = 1600 } = opts;
  const rng = mulberry32(seed);
  const target = makeMoons(rng);

  // 噪声调度
  const T = steps;
  const beta: number[] = [];
  for (let t = 0; t < T; t++) beta.push(1e-4 + (0.06 - 1e-4) * (t / (T - 1)));
  const alpha = beta.map((b) => 1 - b);
  const alphabar: number[] = [];
  alpha.reduce((acc, a, i) => (alphabar[i] = acc * a), 1);

  // 去噪器：输入 [x, y, t/T] -> 预测噪声 [ex, ey]
  const net: MLP = makeMLP(3, 32, 2, rng);
  const lr = 0.02;

  // 训练：随机取 x0、随机 t，预测加进去的噪声
  for (let it = 0; it < trainIters; it++) {
    const g = zeroGrads(net);
    const B = 32;
    for (let b = 0; b < B; b++) {
      const x0 = target[Math.floor(rng() * target.length)];
      const t = Math.floor(rng() * T);
      const ex = gaussian(rng, 0, 1);
      const ey = gaussian(rng, 0, 1);
      const sa = Math.sqrt(alphabar[t]);
      const sb = Math.sqrt(1 - alphabar[t]);
      const xt = [sa * x0.x + sb * ex, sa * x0.y + sb * ey, t / T];
      const f = forward(net, xt);
      backwardAccum(net, xt, f, [f.out[0] - ex, f.out[1] - ey], g); // MSE 梯度
    }
    applyGrads(net, g, lr, 1 / B);
  }

  // 反向采样：从纯噪声一步步去噪，每步记一帧
  const M = 120;
  let pts = Array.from({ length: M }, () => ({ x: gaussian(rng, 0, 1) * 2.2, y: gaussian(rng, 0, 1) * 2.2 }));
  const frames: Frame<ClusterState>[] = [];
  const snapshot = (iter: number, noiseLevel: number) => {
    frames.push({
      iter,
      state: {
        points: [
          ...target.map((p) => ({ x: p.x, y: p.y, cluster: 1 })), // 目标（参考，淡）
          ...pts.map((p) => ({ x: p.x, y: p.y, cluster: 0 })), // 生成中
        ],
        centroids: [],
      },
      metrics: { noiseLevel },
    });
  };
  snapshot(0, 1);

  for (let t = T - 1; t >= 0; t--) {
    const sb = Math.sqrt(1 - alphabar[t]);
    pts = pts.map((p) => {
      const f = forward(net, [p.x, p.y, t / T]);
      // 预测的去噪均值
      const coef = beta[t] / sb;
      const inv = 1 / Math.sqrt(alpha[t]);
      let nx = inv * (p.x - coef * f.out[0]);
      let ny = inv * (p.y - coef * f.out[1]);
      if (t > 0) {
        nx += Math.sqrt(beta[t]) * gaussian(rng, 0, 1) * 0.5;
        ny += Math.sqrt(beta[t]) * gaussian(rng, 0, 1) * 0.5;
      }
      return { x: nx, y: ny };
    });
    snapshot(T - t, sb);
  }

  return {
    meta: {
      id: "diffusion",
      title: "扩散模型 Diffusion",
      family: "clusters",
      algorithm: "Diffusion (DDPM)",
      description: "Stable Diffusion 的底层。从纯噪声一步步去噪，生成目标分布（双月形）。淡蓝=目标参考。",
      tutorial: {
        problem: "怎么从一团随机噪声“雕刻”出有结构的样本？这是当下最主流的生成方式（Stable Diffusion 等）。",
        intuition:
          "训练时不断给真实数据加噪声，并训练一个网络去“预测加了多少噪声”。生成时反过来：从纯噪声出发，用这个网络一步步把噪声去掉，结构就慢慢浮现出来——像从毛玻璃后逐渐显影。",
        watch: [
          "第 0 帧是一团纯噪声（散乱）",
          "随去噪步推进，点云逐渐聚成“双月形”——和淡蓝色的目标分布吻合",
          "右侧 noiseLevel 从 1 降到 0，表示噪声被一步步去掉",
        ],
        concepts: [
          { term: "前向加噪", explain: "训练时给数据逐步加高斯噪声，直到变纯噪声" },
          { term: "去噪网络", explain: "学习“预测当前样本里的噪声”，反向时据此一点点去噪" },
          { term: "采样/反向过程", explain: "从噪声出发，多步去噪生成新样本（这里看到的动画）" },
        ],
        tryThis: "拖时间轴看一团噪声怎么一步步显影成双月形；点“重新生成数据”换随机种子重采样。",
      },
      hyperparams: { T: steps, hidden: 32, trainIters },
    },
    frames,
  };
}
