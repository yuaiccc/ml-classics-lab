// GAN（生成对抗网络）—— 浏览器手写。生成器 vs 判别器博弈：
// 生成器想造出以假乱真的点，判别器想分清真假；对抗训练让生成分布逼近真实分布（一个圆环）。
import { Trajectory, ClusterState, Frame, Point2D } from "@/player/types";
import { mulberry32, gaussian } from "./rng";
import { MLP, makeMLP, forward, zeroGrads, backwardAccum, applyGrads, sigmoid } from "./nn";

export interface GANOptions {
  seed?: number;
  epochs?: number;
}

export function runGAN(opts: GANOptions = {}): Trajectory<ClusterState> {
  const { seed = (Date.now() & 0xffff) >>> 0, epochs = 240 } = opts;
  const rng = mulberry32(seed);
  const M = 100;

  // 真实分布：半径 2.8 的圆环
  const real: Point2D[] = Array.from({ length: M }, () => {
    const a = rng() * Math.PI * 2;
    const r = 2.8 + gaussian(rng, 0, 0.18);
    return { x: r * Math.cos(a), y: r * Math.sin(a) };
  });

  const G: MLP = makeMLP(2, 16, 2, rng); // z(2) -> 2D
  const D: MLP = makeMLP(2, 16, 1, rng); // 2D -> 真假 logit
  const lr = 0.04;

  // 固定一批隐变量用于可视化（让同一批生成点持续演化）
  const zVis = Array.from({ length: M }, () => [gaussian(rng, 0, 1), gaussian(rng, 0, 1)]);

  const frames: Frame<ClusterState>[] = [];
  const snapshot = (iter: number, realness: number) => {
    const fake = zVis.map((z) => forward(G, z).out);
    const points = [
      ...real.map((p) => ({ x: p.x, y: p.y, cluster: 0 })),
      ...fake.map((o) => ({ x: o[0], y: o[1], cluster: 1 })),
    ];
    frames.push({ iter, state: { points, centroids: [] }, metrics: { realness } });
  };

  for (let e = 0; e <= epochs; e++) {
    // 采样一批 z
    const z = Array.from({ length: M }, () => [gaussian(rng, 0, 1), gaussian(rng, 0, 1)]);
    const fake = z.map((zz) => forward(G, zz));

    // —— 训练判别器 D：真→1，假→0 ——
    const gD = zeroGrads(D);
    let realnessSum = 0;
    for (const r of real) {
      const f = forward(D, [r.x, r.y]);
      const p = sigmoid(f.out[0]);
      backwardAccum(D, [r.x, r.y], f, [p - 1], gD);
    }
    for (const fk of fake) {
      const f = forward(D, fk.out);
      const p = sigmoid(f.out[0]);
      realnessSum += p;
      backwardAccum(D, fk.out, f, [p - 0], gD);
    }
    applyGrads(D, gD, lr, 1 / (2 * M));

    // —— 训练生成器 G：想让 D 把假的判成真（非饱和损失）——
    const gG = zeroGrads(G);
    for (let i = 0; i < M; i++) {
      const fk = forward(G, z[i]);
      const fd = forward(D, fk.out);
      const p = sigmoid(fd.out[0]);
      const gDtmp = zeroGrads(D);
      backwardAccum(D, fk.out, fd, [p - 1], gDtmp); // 想要标签1
      backwardAccum(G, z[i], fk, gDtmp.dIn, gG); // 把对输入的梯度回传进 G
    }
    applyGrads(G, gG, lr, 1 / M);

    if (e % 5 === 0) snapshot(e, realnessSum / M);
  }

  return {
    meta: {
      id: "gan",
      title: "GAN 生成对抗网络",
      family: "clusters",
      algorithm: "GAN",
      description: "生成器 vs 判别器博弈。蓝=真实分布(圆环)，绿=生成点，对抗训练让绿点逐渐铺满圆环。",
      tutorial: {
        problem: "怎么让机器“无中生有”造出像真实数据的新样本？GAN：用两个网络互相博弈。",
        intuition:
          "生成器(G)把随机噪声映射成假样本，判别器(D)负责分辨真假。G 努力骗过 D，D 努力识破 G——道高一尺魔高一丈。两者对抗到最后，G 造出的分布就和真实分布几乎一样了。这是 2014 年点燃生成式 AI 的关键思想。",
        watch: [
          "蓝点=真实分布（一个圆环），绿点=生成器造的假点",
          "绿点一开始挤在中间一团，随训练逐渐散开、铺到圆环上",
          "右侧 realness = 判别器对假点打的“像真”分数，越接近 0.5 说明越骗得过",
        ],
        concepts: [
          { term: "生成器 G", explain: "把随机噪声 z 映射成样本，想造得以假乱真" },
          { term: "判别器 D", explain: "二分类器，努力区分真实样本和生成样本" },
          { term: "对抗训练", explain: "G 和 D 此消彼长的博弈，最终 G 逼近真实分布" },
        ],
        tryThis: "拖时间轴看绿点从一团慢慢铺成圆环；GAN 训练不稳，偶尔会“模式坍塌”挤一块，正常现象。",
      },
      hyperparams: { latent: 2, hidden: 16, epochs, lr },
    },
    frames,
  };
}
