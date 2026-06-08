import { Trajectory, Frame, ImageGridState } from "@/player/types";

// 6×6 GridWorld：左上起点，右下目标，每步 -1，到达目标结束。
// Sarsa（on-policy TD）学习 Q，逐帧把状态价值 V(s)=max_a Q 画成热图，
// 观察价值如何从目标格逐步向外扩散（信用分配）。

const SIZE = 6;
const N_STATES = SIZE * SIZE;
const N_ACTIONS = 4; // 0上 1下 2左 3右
const GOAL = N_STATES - 1;

const ALPHA = 0.2;
const GAMMA = 0.95;
const EPS = 0.15;
const EPISODES = 240;
const CAPTURE_EVERY = 10;

function step(s: number, a: number): number {
  let r = Math.floor(s / SIZE);
  let c = s % SIZE;
  if (a === 0) r = Math.max(0, r - 1);
  else if (a === 1) r = Math.min(SIZE - 1, r + 1);
  else if (a === 2) c = Math.max(0, c - 1);
  else c = Math.min(SIZE - 1, c + 1);
  return r * SIZE + c;
}

function epsGreedy(Q: number[][], s: number): number {
  if (Math.random() < EPS) return Math.floor(Math.random() * N_ACTIONS);
  let best = 0;
  for (let a = 1; a < N_ACTIONS; a++) if (Q[s][a] > Q[s][best]) best = a;
  return best;
}

export function runSarsa(): Trajectory {
  const Q: number[][] = Array.from({ length: N_STATES }, () => new Array(N_ACTIONS).fill(0));
  const frames: Frame[] = [];

  const captureFrame = (episode: number): void => {
    // V(s) = max_a Q；归一化到 [0,1]（目标=1 最亮）
    const V = new Array(N_STATES).fill(0).map((_, s) => Math.max(...Q[s]));
    const min = Math.min(...V);
    const max = Math.max(...V);
    const range = max - min || 1;
    const pixels = V.map((v) => Number(((v - min) / range).toFixed(3)));
    const data: ImageGridState = {
      groups: [
        {
          title: `状态价值 V(s) 热图 · 第 ${episode} 回合（亮=价值高，目标在右下）`,
          w: SIZE,
          h: SIZE,
          colormap: "sequential",
          images: [pixels],
        },
      ],
    };
    frames.push({
      iter: episode,
      state: { family: "image-grid", data },
      metrics: { episode, max_value: Number(max.toFixed(3)) },
    });
  };

  captureFrame(0);
  for (let ep = 1; ep <= EPISODES; ep++) {
    let s = 0;
    let a = epsGreedy(Q, s);
    let steps = 0;
    while (s !== GOAL && steps < 200) {
      const sNext = step(s, a);
      const reward = sNext === GOAL ? 0 : -1;
      const aNext = epsGreedy(Q, sNext);
      // Sarsa 更新：用实际下一动作 aNext（on-policy）
      const target = sNext === GOAL ? reward : reward + GAMMA * Q[sNext][aNext];
      Q[s][a] += ALPHA * (target - Q[s][a]);
      s = sNext;
      a = aNext;
      steps++;
    }
    if (ep % CAPTURE_EVERY === 0) captureFrame(ep);
  }

  return {
    meta: {
      id: "sarsa",
      title: "Sarsa 时序差分控制",
      algorithm: "Sarsa",
      category: "rl",
      source: "browser",
      abstract:
        "Sarsa 是 on-policy 的时序差分控制：用『实际采取的下一动作』来更新 Q。在 6×6 GridWorld 上逐回合把状态价值画成热图，观察价值如何从目标格沿可行路径逐步向外扩散——这就是强化学习中的信用分配。",
      description: "6×6 GridWorld 上运行 Sarsa，逐回合导出状态价值 V(s)=max_a Q 的热图，展示价值传播。",
      hyperparams: { alpha: ALPHA, gamma: GAMMA, epsilon: EPS, episodes: EPISODES, grid: `${SIZE}×${SIZE}` },
      insight:
        "Sarsa 与 Q-Learning 的关键区别：Sarsa 用实际下一动作 a' 更新（on-policy，更保守、更安全），Q-Learning 用最大 Q（off-policy，更激进）。在有风险的环境（如悬崖）Sarsa 会学到更稳妥的绕行路径。",
    },
    frames,
  };
}
