// 表格 Q-Learning（GridWorld）—— 最本源的强化学习（DQN 之前）。
// 智能体在格子世界里试错，价值从目标格逐步“反向传播”到全图，贪心策略箭头指向目标。
import { Trajectory, GridWorldState, Frame } from "@/player/types";
import { mulberry32 } from "./rng";

const ROWS = 6;
const COLS = 6;
const ACTIONS = 4; // 0上 1右 2下 3左
const DR = [-1, 0, 1, 0];
const DC = [0, 1, 0, -1];

export interface QLearnOptions {
  seed?: number;
  episodes?: number;
}

export function runQLearning(opts: QLearnOptions = {}): Trajectory<GridWorldState> {
  const { seed = (Date.now() & 0xffff) >>> 0, episodes = 200 } = opts;
  const rng = mulberry32(seed);
  const S = ROWS * COLS;
  const goal = S - 1; // 右下角
  const pit = ROWS * 2 + 2; // 中间偏上的陷阱
  const start = 0;

  const Q: number[][] = Array.from({ length: S }, () => new Array(ACTIONS).fill(0));
  const gamma = 0.95;
  const alpha = 0.5;

  const idx = (r: number, c: number) => r * COLS + c;
  const stepEnv = (s: number, a: number) => {
    const r = Math.floor(s / COLS);
    const c = s % COLS;
    let nr = r + DR[a];
    let nc = c + DC[a];
    if (nr < 0 || nr >= ROWS || nc < 0 || nc >= COLS) {
      nr = r;
      nc = c;
    }
    const ns = idx(nr, nc);
    let reward = -0.04;
    let done = false;
    if (ns === goal) {
      reward = 1;
      done = true;
    } else if (ns === pit) {
      reward = -1;
      done = true;
    }
    return { ns, reward, done };
  };

  const greedy = (s: number) => {
    let best = 0;
    let bv = -Infinity;
    for (let a = 0; a < ACTIONS; a++) if (Q[s][a] > bv) { bv = Q[s][a]; best = a; }
    return best;
  };

  const snapshot = (iter: number): Frame<GridWorldState> => {
    const values = Q.map((row) => Math.max(...row));
    const policy = Q.map((_, s) => (s === goal || s === pit ? -1 : greedy(s)));
    // 贪心 rollout 的回报（衡量学得好不好）
    let s = start;
    let ret = 0;
    for (let t = 0; t < 50; t++) {
      const a = greedy(s);
      const { ns, reward, done } = stepEnv(s, a);
      ret += reward;
      s = ns;
      if (done) break;
    }
    return {
      iter,
      state: { rows: ROWS, cols: COLS, values, policy, goal, pit },
      metrics: { greedyReturn: ret },
    };
  };

  const frames: Frame<GridWorldState>[] = [snapshot(0)];
  for (let ep = 1; ep <= episodes; ep++) {
    let s = start;
    const eps = Math.max(0.05, 1 - ep / (episodes * 0.6));
    for (let t = 0; t < 100; t++) {
      const a = rng() < eps ? Math.floor(rng() * ACTIONS) : greedy(s);
      const { ns, reward, done } = stepEnv(s, a);
      const target = done ? reward : reward + gamma * Math.max(...Q[ns]);
      Q[s][a] += alpha * (target - Q[s][a]);
      s = ns;
      if (done) break;
    }
    if (ep % 5 === 0) frames.push(snapshot(ep));
  }

  return {
    meta: {
      id: "qlearning",
      title: "Q-Learning · 格子世界",
      family: "gridworld",
      algorithm: "Tabular Q-Learning",
      description: "最本源的强化学习：智能体在格子里试错，价值从目标反向扩散，策略箭头逐渐指向目标。",
      tutorial: {
        problem: "没有人教，智能体怎么在迷宫里学会走到终点、避开陷阱？这是强化学习最本源的问题。",
        intuition:
          "给每个“格子+动作”维护一个价值 Q。每走一步就更新：当前价值 ← 即时奖励 + 未来最优价值的折扣。到达终点的高价值会像水波一样，一格一格地“反向扩散”回起点，于是每格的最优方向（箭头）慢慢都指向终点。",
        watch: [
          "价值热图：终点（绿）附近先亮，价值逐渐向四周扩散",
          "每格的箭头 = 当前最优动作，慢慢都指向终点、绕开陷阱（红）",
          "右侧“贪心回报”上升 = 按当前策略能稳定走到终点",
        ],
        concepts: [
          { term: "Q 值", explain: "在某状态采取某动作的长期价值估计" },
          { term: "时序差分(TD)更新", explain: "用“即时奖励+下一步最优价值”来修正当前估计" },
          { term: "价值反向传播", explain: "终点的高价值沿路径一格格回传到起点" },
        ],
        tryThis: "拖时间轴看价值从终点扩散、箭头逐渐成形；这就是 DQN 之前的“表格版”强化学习。",
      },
      hyperparams: { gamma, alpha, episodes, grid: `${ROWS}×${COLS}` },
    },
    frames,
  };
}
