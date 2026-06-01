// 「强化学习之前」的 CartPole 解法：经典控制（状态反馈 / LQR 形式）。
// 不学习、不试错——直接用物理方程算出控制律 u = -K·x，把杆稳住。
// 和 RL 组里的 PPO 对照：结果一样稳，但这个完全没有训练过程。
import { Trajectory, EnvState, Frame } from "@/player/types";
import { step, isTerminal, CartPoleState, OPTIMAL_WEIGHTS } from "@/utils/cartpole";
import { mulberry32 } from "./rng";

export interface ControlOptions {
  seed?: number;
  maxSteps?: number;
}

export function runCartPoleControl(opts: ControlOptions = {}): Trajectory<EnvState> {
  const { seed = (Date.now() & 0xffff) >>> 0, maxSteps = 300 } = opts;
  const rng = mulberry32(seed);

  // 初始给一个明显扰动，好看控制器怎么把它拉回竖直
  let s: CartPoleState = {
    x: (rng() * 2 - 1) * 0.3,
    xDot: (rng() * 2 - 1) * 0.2,
    theta: (rng() * 2 - 1) * 0.18, // ~ ±10°
    thetaDot: (rng() * 2 - 1) * 0.2,
  };

  // 状态反馈增益 K（LQR 解出的那种线性反馈系数；这里用现成的一组）
  const K = OPTIMAL_WEIGHTS;
  const frames: Frame<EnvState>[] = [];

  for (let t = 0; t < maxSteps; t++) {
    // 控制律：u = K·状态。每一项=对“位置/速度/角度/角速度”的加权反馈
    const u = K[0] * s.x + K[1] * s.xDot + K[2] * s.theta + K[3] * s.thetaDot;
    const action = u >= 0 ? 1 : 0; // CartPole 只能左/右推（bang-bang）
    frames.push({
      iter: t,
      state: { observation: [s.x, s.xDot, s.theta, s.thetaDot], action },
      metrics: { angle: Math.abs(s.theta) * (180 / Math.PI) },
    });
    if (isTerminal(s)) break;
    s = step(s, action);
  }

  return {
    meta: {
      id: "cartpole-control",
      title: "CartPole · 经典控制（状态反馈）",
      family: "env",
      algorithm: "State Feedback (LQR-style)",
      envId: "CartPole-v1",
      description:
        "RL 之前的解法：知道物理方程，用状态反馈控制律 u = K·x 直接把杆稳住，不需要任何学习。",
      tutorial: {
        problem: "在没有强化学习的年代，CartPole（倒立摆）是怎么被解决的？",
        intuition:
          "如果你知道小车质量、杆长、重力这些物理参数，倒立摆就是个经典控制问题。控制律是 u = K·x：把“位置、速度、杆角、角速度”四个量各乘一个增益再相加，得出该往哪边推。增益 K 由系统物理特性解出来（LQR 解 Riccati 方程），是算出来的、不是学出来的——零训练、零试错。",
        watch: [
          "初始杆是歪的，控制器几十步内就把它拉回竖直并稳住",
          "右侧“杆偏离角度”曲线从大迅速降到接近 0°",
          "对照「强化学习 · env」里的 CartPole·PPO：结果一样稳，但那个是从奖励里学出来的，这个是从物理算出来的",
        ],
        concepts: [
          { term: "状态反馈 / LQR", explain: "u = -K·x，最优线性控制器；K 由系统动力学解析求出" },
          { term: "增益 K", explain: "对每个状态量的反馈系数，决定“推多大力”" },
          { term: "model-based vs model-free", explain: "经典控制要知道物理方程；RL 不用，只靠奖励试错" },
        ],
        tryThis: "点“重新生成数据”换一个初始扰动，看控制器每次都能拉回；再切到 RL 组的 CartPole·PPO 对比两种思路。",
      },
      hyperparams: { gains_K: `[${K.join(", ")}]`, maxSteps },
    },
    frames,
  };
}
